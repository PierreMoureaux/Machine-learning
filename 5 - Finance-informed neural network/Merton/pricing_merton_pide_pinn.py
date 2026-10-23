import time
import numpy as np
import pandas as pd
import scipy.stats as si
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import math

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Market and Merton Jump Diffusion Parameters
K = 100.0        # Strike Price
r = 0.05         # Risk-free rate
T = 1.0          # Maturity
sigma1 = 0.2     # Volatility of Asset 1
sigma2 = 0.25    # Volatility of Asset 2
rho = 0.3        # Correlation between asset diffusions
lam = 0.1        # Jump intensity (frequency)
mu_j = -0.05     # Mean of log-jump size
sigma_j = 0.1    # Volatility of log-jump size

# Derived parameter for Merton drift correction
# E[Y - 1] where ln(Y) ~ N(mu_j, sigma_j^2)
kappa_j = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0

# Splitting boundaries for generalization tests
S_train_min, S_train_max = 80.0, 120.0
S_extrap_min, S_extrap_max = 50.0, 150.0

# =============================================================================
# 1D MERTON ANALYTICAL SOLUTION (INFINITE SERIES)
# =============================================================================
def black_scholes_call_1d(S, K, T, r, sigma):
    if T <= 1e-5:
        return np.maximum(S - K, 0.0)
    S = np.maximum(S, 1e-5)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)

def merton_call_analytical_1d(S_vec, K, T_vec, r, sigma, lam, mu_j, sigma_j, n_terms=30):
    N = len(S_vec)
    prices = np.zeros(N)
    for i in range(N):
        S, T_mat = S_vec[i], T_vec[i]
        if T_mat <= 1e-5:
            prices[i] = max(S - K, 0.0)
            continue
        
        price_sum = 0.0
        kappa_j_loc = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
        for n in range(n_terms):
            r_n = r - lam * kappa_j_loc + (n * (mu_j + 0.5 * sigma_j**2)) / T_mat
            sigma_n = np.sqrt(sigma**2 + (n * sigma_j**2) / T_mat)
            bs_p = black_scholes_call_1d(S, K, T_mat, r_n, sigma_n)
            
            weight = (np.exp(-lam * T_mat) * (lam * T_mat)**n) / math.factorial(n)
            price_sum += weight * bs_p
        prices[i] = max(price_sum, 0.0)
    return prices

# =============================================================================
# 1D PIDE SOLVER - FINITE DIFFERENCES (IMPLICIT-EXPLICIT IMEX)
# =============================================================================
def pide_solver_merton_1d(S_vec, K, T, r, sigma, lam, mu_j, sigma_j):
    S_max_grid = 300.0
    N, NJ = 300, 200
    dt, dS = T / NJ, S_max_grid / N
    s_grid = np.linspace(0, S_max_grid, N + 1)
    grid = np.maximum(s_grid - K, 0.0)
    
    # Precompute jump density truncation grid weights
    kappa_j_loc = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    y_min, y_max = 0.01, 3.0
    M_y = 100
    y_vals = np.linspace(y_min, y_max, M_y)
    dy = (y_max - y_min) / (M_y - 1)
    
    # Log-normal density function
    f_Y = (1.0 / (y_vals * sigma_j * np.sqrt(2 * np.pi))) * np.exp(-((np.log(y_vals) - mu_j)**2) / (2 * sigma_j**2))
    f_Y /= np.sum(f_Y * dy) # Normalize numerical density
    
    # IMEX Scheme Operator Matrix Setup
    # A_matrix implements diffusion and drift implicitly
    a = np.zeros(N + 1)
    b = np.zeros(N + 1)
    c = np.zeros(N + 1)
    
    drift = r - lam * kappa_j_loc
    for j in range(1, N):
        a[j] = 0.5 * dt * (drift * j - (sigma**2) * (j**2))
        b[j] = 1.0 + dt * ((sigma**2) * (j**2) + r + lam)
        c[j] = -0.5 * dt * (drift * j + (sigma**2) * (j**2))
        
    A = np.zeros((N + 1, N + 1))
    A[0, 0], A[N, N] = 1.0, 1.0
    for j in range(1, N):
        A[j, j-1], A[j, j], A[j, j+1] = a[j], b[j], c[j]
    
    A_inv = np.linalg.inv(A)
    
    for _ in range(NJ):
        # Compute Integral Term Explicitly
        integral_term = np.zeros(N + 1)
        for j in range(1, N):
            S_curr = s_grid[j]
            S_scaled = S_curr * y_vals
            grid_interp = np.interp(S_scaled, s_grid, grid)
            integral_term[j] = lam * np.sum(grid_interp * f_Y * dy) * dt
            
        rhs = grid + integral_term
        rhs[0] = 0.0
        rhs[N] = S_max_grid - K * np.exp(-r * T)
        grid = A_inv.dot(rhs)
        
    return np.interp(S_vec, s_grid, grid)

# =============================================================================
# MONTE CARLO SIMULATOR WITH COMPONENT JUMPS (1D & 2D SYSTEMIC JUMPS)
# =============================================================================
def monte_carlo_merton_1d(S_vec, K, T_vec, r, sigma, lam, mu_j, sigma_j, M=15000, N_steps=100):
    N_assets = len(S_vec)
    prices = np.zeros(N_assets)
    kappa_j_loc = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    
    for i in range(N_assets):
        S0, t_mat = S_vec[i], T_vec[i]
        if t_mat <= 1e-5:
            prices[i] = max(S0 - K, 0.0)
            continue
            
        dt = t_mat / N_steps
        S = np.ones(M) * S0
        
        for _ in range(N_steps):
            Z = np.random.standard_normal(M)
            N_jumps = np.random.poisson(lam * dt, M)
            jump_factor = np.zeros(M)
            for m in range(M):
                if N_jumps[m] > 0:
                    jump_factor[m] = np.sum(np.random.normal(mu_j, sigma_j, N_jumps[m]))
            
            S = S * np.exp((r - lam * kappa_j_loc - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z + jump_factor)
            
        prices[i] = np.exp(-r * t_mat) * np.mean(np.maximum(S - K, 0.0))
    return prices

def monte_carlo_merton_basket_2d(S1_vec, S2_vec, K, T_vec, r, sigma1, sigma2, rho, lam, mu_j, sigma_j, M=15000, N_steps=100):
    N_assets = len(S1_vec)
    prices = np.zeros(N_assets)
    kappa_j_loc = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    
    for i in range(N_assets):
        S1_0, S2_0, t_mat = S1_vec[i], S2_vec[i], T_vec[i]
        if t_mat <= 1e-5:
            prices[i] = max(max(S1_0, S2_0) - K, 0.0)
            continue
            
        dt = t_mat / N_steps
        S1 = np.ones(M) * S1_0
        S2 = np.ones(M) * S2_0
        
        for _ in range(N_steps):
            Z1 = np.random.standard_normal(M)
            Z2 = rho * Z1 + np.sqrt(1.0 - rho**2) * np.random.standard_normal(M)
            
            # Systemic shared jumps representing common macro shocks
            N_jumps = np.random.poisson(lam * dt, M)
            jump_factor = np.zeros(M)
            for m in range(M):
                if N_jumps[m] > 0:
                    jump_factor[m] = np.sum(np.random.normal(mu_j, sigma_j, N_jumps[m]))
                    
            S1 = S1 * np.exp((r - lam * kappa_j_loc - 0.5 * sigma1**2) * dt + sigma1 * np.sqrt(dt) * Z1 + jump_factor)
            S2 = S2 * np.exp((r - lam * kappa_j_loc - 0.5 * sigma2**2) * dt + sigma2 * np.sqrt(dt) * Z2 + jump_factor)
            
        prices[i] = np.exp(-r * t_mat) * np.mean(np.maximum(np.maximum(S1, S2) - K, 0.0))
    return prices

# =============================================================================
# DEEP LEARNING MODEL CLASSES WITH CONSTRAINED ARBITRAGE PROTECTION
# =============================================================================
class MertonNet1D(nn.Module):
    def __init__(self):
        super(MertonNet1D, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1)
        )
    def forward(self, s, t):
        inputs = torch.cat([s / K, t], dim=1)
        out = self.net(inputs)
        return torch.nn.functional.softplus(out) * K

class MertonNet2D(nn.Module):
    def __init__(self):
        super(MertonNet2D, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 1)
        )
    def forward(self, s1, s2, t):
        inputs = torch.cat([s1 / K, s2 / K, t], dim=1)
        out = self.net(inputs)
        return torch.nn.functional.softplus(out) * K

# =============================================================================
# COLOCATION DATASETS AND STRUCTURAL SETUP
# =============================================================================
num_train = 800
S_data_1d = np.random.uniform(S_train_min, S_train_max, (num_train, 1))
t_data_1d = np.random.uniform(0.0, T, (num_train, 1))
T_vec_1d = T - t_data_1d[:, 0]

C_data_1d = merton_call_analytical_1d(S_data_1d[:, 0], K, T_vec_1d, r, sigma1, lam, mu_j, sigma_j).reshape(-1, 1)

S_1d_t = torch.tensor(S_data_1d, dtype=torch.float32)
t_1d_t = torch.tensor(t_data_1d, dtype=torch.float32)
C_1d_t = torch.tensor(C_data_1d, dtype=torch.float32)

S1_data_2d = np.random.uniform(S_train_min, S_train_max, (num_train, 1))
S2_data_2d = np.random.uniform(S_train_min, S_train_max, (num_train, 1))
t_data_2d = np.random.uniform(0.0, T, (num_train, 1))
T_vec_2d = T - t_data_2d[:, 0]

C_data_2d = monte_carlo_merton_basket_2d(S1_data_2d[:, 0], S2_data_2d[:, 0], K, T_vec_2d, r, sigma1, sigma2, rho, lam, mu_j, sigma_j, M=2500).reshape(-1, 1)

S1_2d_t = torch.tensor(S1_data_2d, dtype=torch.float32)
S2_2d_t = torch.tensor(S2_data_2d, dtype=torch.float32)
t_2d_t = torch.tensor(t_data_2d, dtype=torch.float32)
C_2d_t = torch.tensor(C_data_2d, dtype=torch.float32)

nn_1d, pinn_1d = MertonNet1D(), MertonNet1D()
nn_2d, pinn_2d = MertonNet2D(), MertonNet2D()

opt_nn1d = optim.Adam(nn_1d.parameters(), lr=0.005)
opt_pinn1d = optim.Adam(pinn_1d.parameters(), lr=0.003)
opt_nn2d = optim.Adam(nn_2d.parameters(), lr=0.005)
opt_pinn2d = optim.Adam(pinn_2d.parameters(), lr=0.003)

num_pde = 1500
S_pde_1d = torch.tensor(np.random.uniform(S_extrap_min, S_extrap_max, (num_pde, 1)), dtype=torch.float32, requires_grad=True)
t_pde_1d = torch.tensor(np.random.uniform(0.0, T, (num_pde, 1)), dtype=torch.float32, requires_grad=True)

S1_pde_2d = torch.tensor(np.random.uniform(S_extrap_min, S_extrap_max, (num_pde, 1)), dtype=torch.float32, requires_grad=True)
S2_pde_2d = torch.tensor(np.random.uniform(S_extrap_min, S_extrap_max, (num_pde, 1)), dtype=torch.float32, requires_grad=True)
t_pde_2d = torch.tensor(np.random.uniform(0.0, T, (num_pde, 1)), dtype=torch.float32, requires_grad=True)

S_term_1d = torch.tensor(np.random.uniform(S_extrap_min, S_extrap_max, (600, 1)), dtype=torch.float32)
t_term_1d = torch.ones_like(S_term_1d) * T
C_term_1d = torch.clamp(S_term_1d - K, min=0.0)

S1_term_2d = torch.tensor(np.random.uniform(S_extrap_min, S_extrap_max, (600, 1)), dtype=torch.float32)
S2_term_2d = torch.tensor(np.random.uniform(S_extrap_min, S_extrap_max, (600, 1)), dtype=torch.float32)
t_term_2d = torch.ones_like(S1_term_2d) * T
C_term_2d = torch.clamp(torch.max(S1_term_2d, S2_term_2d) - K, min=0.0)

# =============================================================================
# OPTIMIZATION LOOP ENFORCING OPERATOR PHYSICS RESIDUALS
# =============================================================================
print("Training Merton Deep Learning PIDE structures...")

epochs = 1000
for epoch in range(epochs):
    opt_nn1d.zero_grad()
    l_nn1 = nn.MSELoss()(nn_1d(S_1d_t, t_1d_t), C_1d_t)
    l_nn1.backward()
    opt_nn1d.step()
    
    opt_pinn1d.zero_grad()
    l_d1 = nn.MSELoss()(pinn_1d(S_1d_t, t_1d_t), C_1d_t)
    
    pred1 = pinn_1d(S_pde_1d, t_pde_1d)
    dC_ds = torch.autograd.grad(pred1.sum(), S_pde_1d, create_graph=True)[0]
    dC_dt = torch.autograd.grad(pred1.sum(), t_pde_1d, create_graph=True)[0]
    d2C_ds2 = torch.autograd.grad(dC_ds.sum(), S_pde_1d, create_graph=True)[0]
    
    # 1D Non-local operator Taylor-expansion bypass approximation
    # I[C] ~ lam * C + lam * kappa_j * S * dC/dS + 0.5 * lam * E[Y^2 - 2Y + 1] * S^2 * d2C/dS2 - lam * C
    # Let var_y = exp(2*mu_j + sigma_j^2)*(exp(sigma_j^2) - 1)
    # E[Y^2] = exp(2*mu_j + 2*sigma_j^2)
    mom2_y = np.exp(2 * mu_j + 2 * sigma_j**2)
    jump_integral_1d = lam * pred1 + lam * kappa_j * S_pde_1d * dC_ds + 0.5 * lam * (mom2_y - 2 * (kappa_j + 1) + 1) * (S_pde_1d**2) * d2C_ds2 - lam * pred1
    
    pde_res1 = dC_dt + (r - lam * kappa_j) * S_pde_1d * dC_ds + 0.5 * (sigma1**2) * (S_pde_1d**2) * d2C_ds2 - r * pred1 + jump_integral_1d
    l_p1 = torch.mean(pde_res1**2)
    l_t1 = nn.MSELoss()(pinn_1d(S_term_1d, t_term_1d), C_term_1d)
    
    (l_d1 + 0.02 * l_p1 + l_t1).backward()
    opt_pinn1d.step()

    opt_nn2d.zero_grad()
    l_nn2 = nn.MSELoss()(nn_2d(S1_2d_t, S2_2d_t, t_2d_t), C_2d_t)
    l_nn2.backward()
    opt_nn2d.step()
    
    opt_pinn2d.zero_grad()
    l_d2 = nn.MSELoss()(pinn_2d(S1_2d_t, S2_2d_t, t_2d_t), C_2d_t)
    
    pred2 = pinn_2d(S1_pde_2d, S2_pde_2d, t_pde_2d)
    dc_ds1 = torch.autograd.grad(pred2.sum(), S1_pde_2d, create_graph=True)[0]
    dc_ds2 = torch.autograd.grad(pred2.sum(), S2_pde_2d, create_graph=True)[0]
    dc_dt2 = torch.autograd.grad(pred2.sum(), t_pde_2d, create_graph=True)[0]
    d2_ds12 = torch.autograd.grad(dc_ds1.sum(), S1_pde_2d, create_graph=True)[0]
    d2_ds22 = torch.autograd.grad(dc_ds2.sum(), S2_pde_2d, create_graph=True)[0]
    d2_ds1s2 = torch.autograd.grad(dc_ds1.sum(), S2_pde_2d, create_graph=True)[0]
    
    # 2D systemic jump operator Taylor expansion bypass approximation
    jump_integral_2d = (lam * kappa_j * S1_pde_2d * dc_ds1 + lam * kappa_j * S2_pde_2d * dc_ds2 + 
                        0.5 * lam * (mom2_y - 2 * (kappa_j + 1) + 1) * (S1_pde_2d**2) * d2_ds12 + 
                        0.5 * lam * (mom2_y - 2 * (kappa_j + 1) + 1) * (S2_pde_2d**2) * d2_ds22 + 
                        lam * (mom2_y - 2 * (kappa_j + 1) + 1) * S1_pde_2d * S2_pde_2d * d2_ds1s2)
    
    pde_res2 = (dc_dt2 + (r - lam * kappa_j) * S1_pde_2d * dc_ds1 + (r - lam * kappa_j) * S2_pde_2d * dc_ds2 + 
                0.5 * (sigma1**2) * (S1_pde_2d**2) * d2_ds12 + 0.5 * (sigma2**2) * (S2_pde_2d**2) * d2_ds22 + 
                rho * sigma1 * sigma2 * S1_pde_2d * S2_pde_2d * d2_ds1s2 - r * pred2 + jump_integral_2d)
    l_p2 = torch.mean(pde_res2**2)
    l_t2 = nn.MSELoss()(pinn_2d(S1_term_2d, S2_term_2d, t_term_2d), C_term_2d)
    
    (l_d2 + 0.01 * l_p2 + l_t2).backward()
    opt_pinn2d.step()

# =============================================================================
# SYSTEM INFERENCE EVALUATIONS AND PERFORMANCE LOG METRICS
# =============================================================================
S_test = np.linspace(S_extrap_min, S_extrap_max, 40)
T_test_fixed = np.ones_like(S_test) * T
t_test_fixed_t = torch.zeros((len(S_test), 1), dtype=torch.float32)
S_test_t = torch.tensor(S_test.reshape(-1, 1), dtype=torch.float32)

t0 = time.time(); pr_analytical_1d = merton_call_analytical_1d(S_test, K, T_test_fixed, r, sigma1, lam, mu_j, sigma_j); time_an_1d = time.time() - t0
t0 = time.time(); pr_pide_1d = pide_solver_merton_1d(S_test, K, T, r, sigma1, lam, mu_j, sigma_j); time_pide_1d = time.time() - t0

with torch.no_grad():
    t0 = time.time(); pr_nn_1d = nn_1d(S_test_t, t_test_fixed_t).numpy().flatten(); time_nn_1d = time.time() - t0
    t0 = time.time(); pr_pinn_1d = pinn_1d(S_test_t, t_test_fixed_t).numpy().flatten(); time_pinn_1d = time.time() - t0

t0 = time.time(); pr_mc_2d = monte_carlo_merton_basket_2d(S_test, S_test, K, T_test_fixed, r, sigma1, sigma2, rho, lam, mu_j, sigma_j, M=25000); time_mc_2d = time.time() - t0

with torch.no_grad():
    t0 = time.time(); pr_nn_2d = nn_2d(S_test_t, S_test_t, t_test_fixed_t).numpy().flatten(); time_nn_2d = time.time() - t0
    t0 = time.time(); pr_pinn_2d = pinn_2d(S_test_t, S_test_t, t_test_fixed_t).numpy().flatten(); time_pinn_2d = time.time() - t0

# =============================================================================
# TRACING THE MULTI-PANEL STRIPS SYSTEM DASHBOARD
# =============================================================================
print("\n" + "="*75)
print("1D MERTON JUMP DIFFUSION VANILLA CALL RESULTS")
df_1d = pd.DataFrame({'Spot_S': S_test, 'Analytical_Merton': pr_analytical_1d, 'PIDE_Grid_1D': pr_pide_1d, 'NN_Standard_1D': pr_nn_1d, 'PINN_1D': pr_pinn_1d})
print(df_1d.iloc[[0, 10, 20, 30, -1]].to_string(index=False))

print("\n" + "="*75)
print("2D MULTI-ASSET SYSTEMIC MERTON BASKET CALL RESULTS")
df_2d = pd.DataFrame({'S1_S2': S_test, 'Monte_Carlo_2D': pr_mc_2d, 'NN_Standard_2D': pr_nn_2d, 'PINN_2D_Basket': pr_pinn_2d})
print(df_2d.iloc[[0, 10, 20, 30, -1]].to_string(index=False))

fig, axs = plt.subplots(2, 2, figsize=(15, 11))

axs[0, 0].plot(S_test, pr_analytical_1d, 'k-', lw=2.5, label='Merton Analytical Series')
axs[0, 0].plot(S_test, pr_pide_1d, 'o', markersize=4, label='IMEX PIDE Solver Grid', alpha=0.8)
axs[0, 0].plot(S_test, pr_nn_1d, 'r:', lw=2.5, label='Standard NN (Data Only)')
axs[0, 0].plot(S_test, pr_pinn_1d, 'g-.', lw=2.5, label='Physics Informed PINN')
axs[0, 0].axvspan(S_train_min, S_train_max, color='yellow', alpha=0.12, label="Training Sampling Zone")
axs[0, 0].set_title("1D Merton Jump Diffusion Pricing & Extrapolation")
axs[0, 0].set_xlabel("Underlying Asset Price (S)"); axs[0, 0].set_ylabel("Call Premium Value")
axs[0, 0].legend(); axs[0, 0].grid(True)

axs[0, 1].plot(S_test, pr_mc_2d, 'b-', lw=2.5, label='Systemic Jump Monte Carlo Ref')
axs[0, 1].plot(S_test, pr_nn_2d, 'r:', lw=2.5, label='Standard NN (Unconstrained)')
axs[0, 1].plot(S_test, pr_pinn_2d, 'g-.', lw=2.5, label='Multi-Asset PINN Framework')
axs[0, 1].axvspan(S_train_min, S_train_max, color='yellow', alpha=0.12, label="Training Sampling Zone")
axs[0, 1].set_title("2D Merton Systemic Basket Call Extrapolation")
axs[0, 1].set_xlabel("Underlying Asset Prices (S1 = S2)"); axs[0, 1].set_ylabel("Basket Premium Value")
axs[0, 1].legend(); axs[0, 1].grid(True)

axs[1, 0].bar(['Analytical', 'IMEX PIDE', 'Standard NN', 'PINN 1D'], [time_an_1d, time_pide_1d, time_nn_1d, time_pinn_1d], color=['black','orange','red','green'])
axs[1, 0].set_yscale('log')
axs[1, 0].set_title("1D Grid/Network Inference Speed Execution Profile")
axs[1, 0].set_ylabel("Log Scale Duration (Seconds)")
axs[1, 0].grid(True, which="both", ls="--", alpha=0.5)

axs[1, 1].bar(['Monte Carlo 2D', 'Standard NN 2D', 'PINN 2D Basket'], [time_mc_2d, time_nn_2d, time_pinn_2d], color=['blue','red','green'])
axs[1, 1].set_yscale('log')
axs[1, 1].set_title("2D Basket Execution Performance Profiles")
axs[1, 1].set_ylabel("Log Scale Duration (Seconds)")
axs[1, 1].grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.show()
