import time
import numpy as np
import pandas as pd
import scipy.stats as si
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(42)
torch.manual_seed(42)

K, r, sigma1, sigma2, rho, T = 100.0, 0.05, 0.2, 0.25, 0.3, 1.0
S_train_min, S_train_max = 80.0, 120.0
S_extrap_min, S_extrap_max = 50.0, 150.0

def black_scholes_call_1d(S, K, T, r, sigma):
    S = np.maximum(S, 1e-5)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)

def finite_difference_1d(S_vec, K, T, r, sigma):
    S_max_grid = 300.0
    N, NJ = 300, 200
    dt, dS = T / NJ, S_max_grid / N
    s_grid = np.linspace(0, S_max_grid, N + 1)
    old_grid = np.maximum(s_grid - K, 0)
    a, b, c = np.zeros(N + 1), np.zeros(N + 1), np.zeros(N + 1)
    for j in range(1, N):
        a[j] = 0.5 * dt * (r * j - sigma**2 * j**2)
        b[j] = 1.0 + dt * (sigma**2 * j**2 + r)
        c[j] = -0.5 * dt * (r * j + sigma**2 * j**2)
    b_bc = np.copy(b)
    b_bc[0], b_bc[N] = 1.0, 1.0
    A = np.zeros((N + 1, N + 1))
    A[0, 0], A[N, N] = 1.0, 1.0
    for j in range(1, N):
        A[j, j-1], A[j, j], A[j, j+1] = a[j], b_bc[j], c[j]
    A_inv = np.linalg.inv(A)
    for _ in range(NJ):
        old_grid[0] = 0.0
        old_grid[N] = S_max_grid - K * np.exp(-r * T)
        old_grid = A_inv.dot(old_grid)
    return np.interp(S_vec, s_grid, old_grid)

class Net1D(nn.Module):
    def __init__(self):
        super(Net1D, self).__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
    def forward(self, s, t):
        return self.net(torch.cat([s/K, t], dim=1)) * K

num_train = 1000
S_data_1d = np.random.uniform(S_train_min, S_train_max, (num_train, 1))
t_data_1d = np.random.uniform(0, T, (num_train, 1))
C_data_1d = black_scholes_call_1d(S_data_1d[:, 0], K, T - t_data_1d[:, 0], r, sigma1).reshape(-1, 1)

S_data_1d_t = torch.tensor(S_data_1d, dtype=torch.float32)
t_data_1d_t = torch.tensor(t_data_1d, dtype=torch.float32)
C_data_1d_t = torch.tensor(C_data_1d, dtype=torch.float32)

nn_1d, pinn_1d = Net1D(), Net1D()
optimizer_nn1d = optim.Adam(nn_1d.parameters(), lr=0.005)
optimizer_pinn1d = optim.Adam(pinn_1d.parameters(), lr=0.003)

S_pde_1d = torch.tensor(np.random.uniform(S_extrap_min, S_extrap_max, (2000, 1)), dtype=torch.float32, requires_grad=True)
t_pde_1d = torch.tensor(np.random.uniform(0, T, (2000, 1)), dtype=torch.float32, requires_grad=True)
S_term_1d = torch.tensor(np.random.uniform(S_extrap_min, S_extrap_max, (800, 1)), dtype=torch.float32)
t_term_1d = torch.ones_like(S_term_1d) * T
C_term_1d = torch.clamp(S_term_1d - K, min=0.0)

print("Training 1D Models...")
for epoch in range(1200):
    optimizer_nn1d.zero_grad()
    loss_nn = nn.MSELoss()(nn_1d(S_data_1d_t, t_data_1d_t), C_data_1d_t)
    loss_nn.backward()
    optimizer_nn1d.step()
    
    optimizer_pinn1d.zero_grad()
    loss_d = nn.MSELoss()(pinn_1d(S_data_1d_t, t_data_1d_t), C_data_1d_t)
    pred_pde = pinn_1d(S_pde_1d, t_pde_1d)
    dC_ds = torch.autograd.grad(pred_pde.sum(), S_pde_1d, create_graph=True)[0]
    dC_dt = torch.autograd.grad(pred_pde.sum(), t_pde_1d, create_graph=True)[0]
    d2C_ds2 = torch.autograd.grad(dC_ds.sum(), S_pde_1d, create_graph=True)[0]
    pde_res = dC_dt + 0.5 * (sigma1**2) * (S_pde_1d**2) * d2C_ds2 + r * S_pde_1d * dC_ds - r * pred_pde
    loss_p = torch.mean(pde_res**2)
    loss_t = nn.MSELoss()(pinn_1d(S_term_1d, t_term_1d), C_term_1d)
    (loss_d + 0.1 * loss_p + loss_t).backward()
    optimizer_pinn1d.step()

def monte_carlo_basket_call(S1_vec, S2_vec, K, T_vec, r, sigma1, sigma2, rho, M=25000):
    N = len(S1_vec)
    prices = np.zeros(N)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov)
    for i in range(N):
        S1, S2, t_mat = S1_vec[i], S2_vec[i], T_vec[i]
        if t_mat <= 1e-5:
            prices[i] = max(max(S1, S2) - K, 0)
            continue
        Z = L.dot(np.random.standard_normal((2, M)))
        S1_T = S1 * np.exp((r - 0.5 * sigma1**2) * t_mat + sigma1 * np.sqrt(t_mat) * Z[0])
        S2_T = S2 * np.exp((r - 0.5 * sigma2**2) * t_mat + sigma2 * np.sqrt(t_mat) * Z[1])
        prices[i] = np.exp(-r * t_mat) * np.mean(np.maximum(np.maximum(S1_T, S2_T) - K, 0))
    return prices

def finite_difference_2d(S1_vec, S2_vec, K, T, r, sigma1, sigma2, rho):
    S_max = 200.0
    N, NJ = 40, 2500  
    dS, dt = S_max / N, T / NJ
    grid = np.zeros((N+1, N+1))
    s_vals = np.linspace(0, S_max, N+1)
    for i in range(N+1):
        for j in range(N+1):
            grid[i, j] = max(max(s_vals[i], s_vals[j]) - K, 0)
    for _ in range(NJ):
        next_grid = np.copy(grid)
        for i in range(1, N):
            for j in range(1, N):
                s1, s2 = s_vals[i], s_vals[j]
                df_ds1 = (grid[i+1, j] - grid[i-1, j]) / (2 * dS)
                df_ds2 = (grid[i, j+1] - grid[i, j-1]) / (2 * dS)
                d2f_ds12 = (grid[i+1, j] - 2 * grid[i, j] + grid[i-1, j]) / (dS**2)
                d2f_ds22 = (grid[i, j+1] - 2 * grid[i, j] + grid[i, j-1]) / (dS**2)
                d2f_ds1s2 = (grid[i+1, j+1] - grid[i+1, j-1] - grid[i-1, j+1] + grid[i-1, j-1]) / (4 * dS**2)
                pde_val = (r * s1 * df_ds1 + r * s2 * df_ds2 + 0.5 * sigma1**2 * s1**2 * d2f_ds12 + 0.5 * sigma2**2 * s2**2 * d2f_ds22 + rho * sigma1 * sigma2 * s1 * s2 * d2f_ds1s2 - r * grid[i, j])
                next_grid[i, j] = grid[i, j] + dt * pde_val
        grid = next_grid
    from scipy.interpolate import RegularGridInterpolator
    interp_func = RegularGridInterpolator((s_vals, s_vals), grid, bounds_error=False, fill_value=None)
    return interp_func(np.vstack((S1_vec, S2_vec)).T)

class Net2D(nn.Module):
    def __init__(self):
        super(Net2D, self).__init__()
        self.net = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1))
    def forward(self, s1, s2, t):
        return self.net(torch.cat([s1/K, s2/K, t], dim=1)) * K

S1_data_2d = np.random.uniform(S_train_min, S_train_max, (num_train, 1))
S2_data_2d = np.random.uniform(S_train_min, S_train_max, (num_train, 1))
t_data_2d = np.random.uniform(0, T, (num_train, 1))
C_data_2d = monte_carlo_basket_call(S1_data_2d[:, 0], S2_data_2d[:, 0], K, T - t_data_2d[:, 0], r, sigma1, sigma2, rho, M=4000).reshape(-1, 1)

S1_data_2d_t = torch.tensor(S1_data_2d, dtype=torch.float32)
S2_data_2d_t = torch.tensor(S2_data_2d, dtype=torch.float32)
t_data_2d_t = torch.tensor(t_data_2d, dtype=torch.float32)
C_data_2d_t = torch.tensor(C_data_2d, dtype=torch.float32)

nn_2d, pinn_2d = Net2D(), Net2D()
optimizer_nn2d = optim.Adam(nn_2d.parameters(), lr=0.005)
optimizer_pinn2d = optim.Adam(pinn_2d.parameters(), lr=0.002)

num_pde_2d = 4000
S1_pde_2d = torch.tensor(np.random.uniform(S_extrap_min, S_extrap_max, (num_pde_2d, 1)), dtype=torch.float32, requires_grad=True)
S2_pde_2d = torch.tensor(np.random.uniform(S_extrap_min, S_extrap_max, (num_pde_2d, 1)), dtype=torch.float32, requires_grad=True)
t_pde_2d = torch.tensor(np.random.uniform(0, T, (num_pde_2d, 1)), dtype=torch.float32, requires_grad=True)

S1_term_2d = torch.tensor(np.random.uniform(S_extrap_min, S_extrap_max, (1200, 1)), dtype=torch.float32)
S2_term_2d = torch.tensor(np.random.uniform(S_extrap_min, S_extrap_max, (1200, 1)), dtype=torch.float32)
t_term_2d = torch.ones_like(S1_term_2d) * T
C_terminal_2d = torch.clamp(torch.max(S1_term_2d, S2_term_2d) - K, min=0.0)

print("Training 2D Models...")
for epoch in range(2500):
    optimizer_nn2d.zero_grad()
    loss_nn2 = nn.MSELoss()(nn_2d(S1_data_2d_t, S2_data_2d_t, t_data_2d_t), C_data_2d_t)
    loss_nn2.backward()
    optimizer_nn2d.step()
    
    optimizer_pinn2d.zero_grad()
    loss_d2 = nn.MSELoss()(pinn_2d(S1_data_2d_t, S2_data_2d_t, t_data_2d_t), C_data_2d_t)
    pred2 = pinn_2d(S1_pde_2d, S2_pde_2d, t_pde_2d)
    dC_ds1 = torch.autograd.grad(pred2.sum(), S1_pde_2d, create_graph=True)[0]
    dC_ds2 = torch.autograd.grad(pred2.sum(), S2_pde_2d, create_graph=True)[0]
    dC_dt2 = torch.autograd.grad(pred2.sum(), t_pde_2d, create_graph=True)[0]
    d2C_ds12 = torch.autograd.grad(dC_ds1.sum(), S1_pde_2d, create_graph=True)[0]
    d2C_ds22 = torch.autograd.grad(dC_ds2.sum(), S2_pde_2d, create_graph=True)[0]
    d2C_ds1s2 = torch.autograd.grad(dC_ds1.sum(), S2_pde_2d, create_graph=True)[0]
    pde_res2 = (dC_dt2 + r*S1_pde_2d*dC_ds1 + r*S2_pde_2d*dC_ds2 + 0.5*(sigma1**2)*(S1_pde_2d**2)*d2C_ds12 + 0.5*(sigma2**2)*(S2_pde_2d**2)*d2C_ds22 + rho*sigma1*sigma2*S1_pde_2d*S2_pde_2d*d2C_ds1s2 - r*pred2)
    loss_p2 = torch.mean(pde_res2**2)
    loss_t2 = nn.MSELoss()(pinn_2d(S1_term_2d, S2_term_2d, t_term_2d), C_terminal_2d)
    (1.5 * loss_d2 + 0.02 * loss_p2 + 1.2 * loss_t2).backward()
    optimizer_pinn2d.step()

S_test = np.linspace(S_extrap_min, S_extrap_max, 40)
S_test_t = torch.tensor(S_test.reshape(-1, 1), dtype=torch.float32)
t_test_t = torch.zeros_like(S_test_t)
T_test_vector = np.ones(len(S_test)) * T

t0 = time.time(); pr_bs_1d = black_scholes_call_1d(S_test, K, T, r, sigma1); time_bs_1d = time.time() - t0
t0 = time.time(); pr_edp_1d = finite_difference_1d(S_test, K, T, r, sigma1); time_edp_1d = time.time() - t0
with torch.no_grad():
    t0 = time.time(); pr_nn_1d = nn_1d(S_test_t, t_test_t).numpy().flatten(); time_nn_1d = time.time() - t0
    t0 = time.time(); pr_pinn_1d = pinn_1d(S_test_t, t_test_t).numpy().flatten(); time_pinn_1d = time.time() - t0

t0 = time.time(); pr_mc_2d = monte_carlo_basket_call(S_test, S_test, K, T_test_vector, r, sigma1, sigma2, rho, M=40000); time_mc_2d = time.time() - t0
t0 = time.time(); pr_edp_2d = finite_difference_2d(S_test, S_test, K, T, r, sigma1, sigma2, rho); time_edp_2d = time.time() - t0
with torch.no_grad():
    t0 = time.time(); pr_nn_2d = nn_2d(S_test_t, S_test_t, t_test_t).numpy().flatten(); time_nn_2d = time.time() - t0
    t0 = time.time(); pr_pinn_2d = pinn_2d(S_test_t, S_test_t, t_test_t).numpy().flatten(); time_pinn_2d = time.time() - t0

print("\n" + "="*78)
print("--- 1D OPTION PRICING RESULTS ---")
df_1d = pd.DataFrame({'Spot S': S_test, 'Analytical BS': pr_bs_1d, 'EDP 1D': pr_edp_1d, 'NN 1D': pr_nn_1d, 'PINN 1D': pr_pinn_1d})
print(df_1d.iloc[[0, 10, 20, 30, -1]].to_string(index=False))

print("\n" + "="*78)
print("--- 2D OPTION PRICING RESULTS ---")
df_2d = pd.DataFrame({'S1 (=S2)': S_test, 'Monte Carlo Ref': pr_mc_2d, 'EDP 2D': pr_edp_2d, 'NN 2D': pr_nn_2d, 'PINN 2D': pr_pinn_2d})
print(df_2d.iloc[[0, 10, 20, 30, -1]].to_string(index=False))

fig, axs = plt.subplots(2, 2, figsize=(16, 12))
axs[0, 0].plot(S_test, pr_bs_1d, 'k-', lw=2.5, label='BS Closed Form')
axs[0, 0].plot(S_test, pr_edp_1d, 'o', markersize=4, label='EDP 1D', alpha=0.8)
axs[0, 0].plot(S_test, pr_nn_1d, 'r:', lw=2.5, label='NN Standard')
axs[0, 0].plot(S_test, pr_pinn_1d, 'g-.', lw=2.5, label='PINN 1D')
axs[0, 0].axvspan(S_train_min, S_train_max, color='yellow', alpha=0.15, label="Training Range")
axs[0, 0].set_title("1D Pricing & Extrapolation")
axs[0, 0].legend(); axs[0, 0].grid(True)

axs[0, 1].plot(S_test, pr_mc_2d, 'k-', lw=2.5, label='Monte Carlo Reference')
axs[0, 1].plot(S_test, pr_edp_2d, 'o', markersize=4, label='EDP 2D', alpha=0.8)
axs[0, 1].plot(S_test, pr_nn_2d, 'r:', lw=2.5, label='NN Standard')
axs[0, 1].plot(S_test, pr_pinn_2d, 'g-.', lw=2.5, label='PINN 2D')
axs[0, 1].axvspan(S_train_min, S_train_max, color='yellow', alpha=0.15, label="Training Range")
axs[0, 1].set_title("2D Basket Call Pricing")
axs[0, 1].legend(); axs[0, 1].grid(True)

axs[1, 0].bar(['Analytical', 'EDP 1D', 'NN 1D', 'PINN 1D'], [time_bs_1d, time_edp_1d, time_nn_1d, time_pinn_1d], color=['black','orange','red','green'])
axs[1, 0].set_yscale('log')

axs[1, 1].bar(['Monte Carlo', 'EDP 2D', 'NN Standard', 'PINN'], [time_mc_2d, time_edp_2d, time_nn_2d, time_pinn_2d], color=['blue','orange','red','green'])
axs[1, 1].set_yscale('log')

plt.tight_layout()
plt.show()
