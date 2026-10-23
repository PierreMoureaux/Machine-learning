import time
import numpy as np
import pandas as pd
import scipy.stats as si
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Market and Bates (Heston + Merton Jumps) Parameters
K = 100.0        # Strike Price
r = 0.05         # Risk-free rate
T = 1.0          # Maturity
kappa = 2.0      # Heston: Mean reversion speed of variance
theta = 0.04     # Heston: Long-term mean of variance
sigma_v = 0.3    # Heston: Vol-of-vol (volatility of variance)
rho = -0.7       # Heston: Correlation between asset and its variance
sigma1 = 0.2     # Asset 1 volatility multiplier
sigma2 = 0.25    # Asset 2 volatility multiplier
rho_assets = 0.5 # Correlation between asset 1 and asset 2 (for Basket)
rho_vars = 0.2   # Correlation between variance 1 and variance 2 (for Basket)

lam = 0.1        # Merton: Jump intensity (frequency)
mu_j = -0.05     # Merton: Mean of log-jump size
sigma_j = 0.1    # Merton: Volatility of log-jump size

# Derived parameter for Bates drift correction
kappa_j = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0

# Splitting boundaries for generalization tests
S_train_min, S_train_max = 80.0, 120.0
S_extrap_min, S_extrap_max = 50.0, 150.0
V_min, V_max = 0.01, 0.09 # Operational variance boundaries

# =============================================================================
# 1D BATES ANALYTICAL SOLUTION (FOURIER INVERSION VIA CARR-MADAN ALPHA METHODS)
# =============================================================================
def bates_char_func(u, S0, V0, T_mat, r, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j):
    # Heston component characteristic function elements
    alpha_h = -0.5 * u * (u + 1j)
    beta_h = kappa - rho * sigma_v * u * 1j
    gamma_h = 0.5 * sigma_v**2
    
    d = np.sqrt(beta_h**2 - 4 * alpha_h * gamma_h)
    g = (beta_h - d) / (beta_h + d)
    
    # Drift corrector for jumps
    w = r - lam * (np.exp(mu_j + 0.5 * sigma_j**2) - 1.0)
    
    C_heston = (w * u * 1j * T_mat + (kappa * theta / sigma_v**2) * 
                ((beta_h - d) * T_mat - 2 * np.log((1 - g * np.exp(-d * T_mat)) / (1 - g))))
    D_heston = ((beta_h - d) / sigma_v**2) * ((1 - np.exp(-d * T_mat)) / (1 - g * np.exp(-d * T_mat)))
    
    # Merton jump component characteristic function extension
    jump_part = lam * T_mat * (np.exp(1j * u * mu_j - 0.5 * u**2 * sigma_j**2) - 1.0)
    
    return np.exp(C_heston + D_heston * V0 + jump_part + 1j * u * np.log(S0))

def bates_call_analytical_single(S0, K, V0, T_mat, r, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j):
    if T_mat <= 1e-5:
        return max(S0 - K, 0.0)
        
    alpha = 0.75 # Carr-Madan damping parameter
    
    def integrand(u):
        psi_u = bates_char_func(u - (alpha + 1) * 1j, S0, V0, T_mat, r, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j)
        denom = (alpha + 1j * u) * (alpha + 1 + 1j * u)
        return np.real(np.exp(-1j * u * np.log(K)) * psi_u / denom)
        
    int_val, _ = integrate.quad(integrand, 0.0, 100.0, limit=150)
    price = (np.exp(-alpha * np.log(K)) / np.pi) * int_val
    return max(price, 0.0)

def bates_call_analytical(S_vec, K, V_vec, T_vec, r, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j):
    N = len(S_vec)
    prices = np.zeros(N)
    for i in range(N):
        prices[i] = bates_call_analytical_single(S_vec[i], K, V_vec[i], T_vec[i], r, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j)
    return prices

# =============================================================================
# BATES MONTE CARLO SIMULATORS (FULL TRUNCATION VECTOR STRUCTURES)
# =============================================================================
def monte_carlo_bates_1d(S_vec, V_vec, K, T_vec, r, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j, M=15000, N_steps=100):
    N_assets = len(S_vec)
    prices = np.zeros(N_assets)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov)
    
    for i in range(N_assets):
        S0, V0, t_mat = S_vec[i], V_vec[i], T_vec[i]
        if t_mat <= 1e-5:
            prices[i] = max(S0 - K, 0.0)
            continue
            
        dt = t_mat / N_steps
        S = np.ones(M) * S0
        V = np.ones(M) * V0
        
        for _ in range(N_steps):
            Z_norm = L.dot(np.random.standard_normal((2, M)))
            N_jumps = np.random.poisson(lam * dt, M)
            jump_factor = np.zeros(M)
            for m in range(M):
                if N_jumps[m] > 0:
                    jump_factor[m] = np.sum(np.random.normal(mu_j, sigma_j, N_jumps[m]))
            
            V_pos = np.maximum(V, 0.0)
            S = S * np.exp((r - lam * kappa_j - 0.5 * V_pos) * dt + np.sqrt(V_pos * dt) * Z_norm[0] + jump_factor)
            V = V + kappa * (theta - V_pos) * dt + sigma_v * np.sqrt(V_pos * dt) * Z_norm[1]
            
        prices[i] = np.exp(-r * t_mat) * np.mean(np.maximum(S - K, 0.0))
    return prices

def monte_carlo_bates_basket_2d(S1_vec, S2_vec, V1_vec, V2_vec, K, T_vec, r, kappa, theta, sigma_v, rho, rho_assets, rho_vars, lam, mu_j, sigma_j, M=15000, N_steps=100):
    N_assets = len(S1_vec)
    prices = np.zeros(N_assets)
    cov = np.array([
        [1.0,        rho,        rho_assets, 0.0],
        [rho,        1.0,        0.0,        rho_vars],
        [rho_assets, 0.0,        1.0,        rho],
        [0.0,        rho_vars,   rho,        1.0]
    ])
    
    eigvals, eigvecs = np.linalg.eigh(cov)
    if np.any(eigvals <= 0):
        eigvals = np.maximum(eigvals, 1e-5)
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
        D = np.diag(1.0 / np.sqrt(np.diag(cov)))
        cov = D @ cov @ D
        
    L = np.linalg.cholesky(cov)
    
    for i in range(N_assets):
        S1_0, S2_0 = S1_vec[i], S2_vec[i]
        V1_0, V2_0 = V1_vec[i], V2_vec[i]
        t_mat = T_vec[i]
        
        if t_mat <= 1e-5:
            prices[i] = max(max(S1_0, S2_0) - K, 0.0)
            continue
            
        dt = t_mat / N_steps
        S1, S2 = np.ones(M) * S1_0, np.ones(M) * S2_0
        V1, V2 = np.ones(M) * V1_0, np.ones(M) * V2_0
        
        for _ in range(N_steps):
            Z = L.dot(np.random.standard_normal((4, M)))
            N_jumps = np.random.poisson(lam * dt, M)
            jump_factor = np.zeros(M)
            for m in range(M):
                if N_jumps[m] > 0:
                    jump_factor[m] = np.sum(np.random.normal(mu_j, sigma_j, N_jumps[m]))
            
            V1_pos = np.maximum(V1, 0.0)
            V2_pos = np.maximum(V2, 0.0)
            
            S1 = S1 * np.exp((r - lam * kappa_j - 0.5 * V1_pos) * dt + np.sqrt(V1_pos * dt) * Z[0] + jump_factor)
            V1 = V1 + kappa * (theta - V1_pos) * dt + sigma_v * np.sqrt(V1_pos * dt) * Z[1]
            
            S2 = S2 * np.exp((r - lam * kappa_j - 0.5 * V2_pos) * dt + np.sqrt(V2_pos * dt) * Z[2] + jump_factor)
            V2 = V2 + kappa * (theta - V2_pos) * dt + sigma_v * np.sqrt(V2_pos * dt) * Z[3]
            
        prices[i] = np.exp(-r * t_mat) * np.mean(np.maximum(np.maximum(S1, S2) - K, 0.0))
    return prices

# =============================================================================
# DEEP LEARNING MODEL CLASSES WITH CONSTRAINED ARBITRAGE PROTECTION
# =============================================================================
class BatesNet1D(nn.Module):
    def __init__(self):
        super(BatesNet1D, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1)
        )
    def forward(self, s, v, t):
        inputs = torch.cat([s / K, v / theta, t], dim=1)
        out = self.net(inputs)
        return torch.nn.functional.softplus(out) * K

class BatesNet2D(nn.Module):
    def __init__(self):
        super(BatesNet2D, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 1)
        )
    def forward(self, s1, s2, v1, v2, t):
        inputs = torch.cat([s1 / K, s2 / K, v1 / theta, v2 / theta, t], dim=1)
        out = self.net(inputs)
        return torch.nn.functional.softplus(out) * K

# =============================================================================
# DATA COLOCATION SETUP
# =============================================================================
num_train = 400
S1_data_1d = np.random.uniform(S_train_min, S_train_max, (num_train, 1))
V1_data_1d = np.random.uniform(V_min, V_max, (num_train, 1))
t_data_1d = np.random.uniform(0.0, T, (num_train, 1))
T_vec_1d = T - t_data_1d[:, 0]

C_data_1d = bates_call_analytical(S1_data_1d[:, 0], K, V1_data_1d[:, 0], T_vec_1d, r, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j).reshape(-1, 1)

S1_1d_t = torch.tensor(S1_data_1d, dtype=torch.float32)
V1_1d_t = torch.tensor(V1_data_1d, dtype=torch.float32)
t_1d_t = torch.tensor(t_data_1d, dtype=torch.float32)
C_1d_t = torch.tensor(C_data_1d, dtype=torch.float32)

S1_data_2d = np.random.uniform(S_train_min, S_train_max, (num_train, 1))
S2_data_2d = np.random.uniform(S_train_min, S_train_max, (num_train, 1))
V1_data_2d = np.random.uniform(V_min, V_max, (num_train, 1))
V2_data_2d = np.random.uniform(V_min, V_max, (num_train, 1))
t_data_2d = np.random.uniform(0.0, T, (num_train, 1))
T_vec_2d = T - t_data_2d[:, 0]

C_data_2d = monte_carlo_bates_basket_2d(
    S1_data_2d[:, 0], S2_data_2d[:, 0], V1_data_2d[:, 0], V2_data_2d[:, 0],
    K, T_vec_2d, r, kappa, theta, sigma_v, rho, rho_assets, rho_vars, lam, mu_j, sigma_j, M=1500
).reshape(-1, 1)

S1_2d_t = torch.tensor(S1_data_2d, dtype=torch.float32)
S2_2d_t = torch.tensor(S2_data_2d, dtype=torch.float32)
V1_2d_t = torch.tensor(V1_data_2d, dtype=torch.float32)
V2_2d_t = torch.tensor(V2_data_2d, dtype=torch.float32)
t_2d_t = torch.tensor(t_data_2d, dtype=torch.float32)
C_2d_t = torch.tensor(C_data_2d, dtype=torch.float32)

nn_1d, pinn_1d = BatesNet1D(), BatesNet1D()
nn_2d, pinn_2d = BatesNet2D(), BatesNet2D()

opt_nn1d = optim.Adam(nn_1d.parameters(), lr=0.005)
opt_pinn1d = optim.Adam(pinn_1d.parameters(), lr=0.003)
opt_nn2d = optim.Adam(nn_2d.parameters(), lr=0.005)
opt_pinn2d = optim.Adam(pinn_2d.parameters(), lr=0.003)

num_pde = 800
S1_pde_1d = torch.tensor(np.random.uniform(S_extrap_min, S_extrap_max, (num_pde, 1)), dtype=torch.float32, requires_grad=True)
V1_pde_1d = torch.tensor(np.random.uniform(V_min, V_max, (num_pde, 1)), dtype=torch.float32, requires_grad=True)
t_pde_1d = torch.tensor(np.random.uniform(0.0, T, (num_pde, 1)), dtype=torch.float32, requires_grad=True)

S1_pde_2d = torch.tensor(np.random.uniform(S_extrap_min, S_extrap_max, (num_pde, 1)), dtype=torch.float32, requires_grad=True)
S2_pde_2d = torch.tensor(np.random.uniform(S_extrap_min, S_extrap_max, (num_pde, 1)), dtype=torch.float32, requires_grad=True)
V1_pde_2d = torch.tensor(np.random.uniform(V_min, V_max, (num_pde, 1)), dtype=torch.float32, requires_grad=True)
V2_pde_2d = torch.tensor(np.random.uniform(V_min, V_max, (num_pde, 1)), dtype=torch.float32, requires_grad=True)
t_pde_2d = torch.tensor(np.random.uniform(0.0, T, (num_pde, 1)), dtype=torch.float32, requires_grad=True)

S_term_1d = torch.tensor(np.random.uniform(S_extrap_min, S_extrap_max, (400, 1)), dtype=torch.float32)
V_term_1d = torch.tensor(np.random.uniform(V_min, V_max, (400, 1)), dtype=torch.float32)
t_term_1d = torch.ones_like(S_term_1d) * T
C_term_1d = torch.clamp(S_term_1d - K, min=0.0)

S1_term_2d = torch.tensor(np.random.uniform(S_extrap_min, S_extrap_max, (400, 1)), dtype=torch.float32)
S2_term_2d = torch.tensor(np.random.uniform(S_extrap_min, S_extrap_max, (400, 1)), dtype=torch.float32)
V1_term_2d = torch.tensor(np.random.uniform(V_min, V_max, (400, 1)), dtype=torch.float32)
V2_term_2d = torch.tensor(np.random.uniform(V_min, V_max, (400, 1)), dtype=torch.float32)
t_term_2d = torch.ones_like(S1_term_2d) * T
C_term_2d = torch.clamp(torch.max(S1_term_2d, S2_term_2d) - K, min=0.0)

# =============================================================================
# REGULARIZATION GRADIENT OPTIMIZATION
# =============================================================================
print("Training Bates Integrated PIDE Structures...")

epochs = 400
for epoch in range(epochs):
    opt_nn1d.zero_grad()
    l_nn1 = nn.MSELoss()(nn_1d(S1_1d_t, V1_1d_t, t_1d_t), C_1d_t)
    l_nn1.backward()
    opt_nn1d.step()
    
    opt_pinn1d.zero_grad()
    l_d1 = nn.MSELoss()(pinn_1d(S1_1d_t, V1_1d_t, t_1d_t), C_1d_t)
    
    pred1 = pinn_1d(S1_pde_1d, V1_pde_1d, t_pde_1d)
    dC_ds1 = torch.autograd.grad(pred1.sum(), S1_pde_1d, create_graph=True)[0]
    dC_dv1 = torch.autograd.grad(pred1.sum(), V1_pde_1d, create_graph=True)[0]
    dC_dt1 = torch.autograd.grad(pred1.sum(), t_pde_1d, create_graph=True)[0]
    d2C_ds12 = torch.autograd.grad(dC_ds1.sum(), S1_pde_1d, create_graph=True)[0]
    d2C_dv12 = torch.autograd.grad(dC_dv1.sum(), V1_pde_1d, create_graph=True)[0]
    d2C_ds1v1 = torch.autograd.grad(dC_ds1.sum(), V1_pde_1d, create_graph=True)[0]
    
    mom2_y = np.exp(2 * mu_j + 2 * sigma_j**2)
    jump_integral_1d = lam * kappa_j * S1_pde_1d * dC_ds1 + 0.5 * lam * (mom2_y - 2 * (kappa_j + 1) + 1) * (S1_pde_1d**2) * d2C_ds12
    
    pde_res1 = (dC_dt1 + (r - lam * kappa_j) * S1_pde_1d * dC_ds1 + kappa * (theta - V1_pde_1d) * dC_dv1 + 
                0.5 * V1_pde_1d * (S1_pde_1d**2) * d2C_ds12 + 0.5 * (sigma_v**2) * V1_pde_1d * d2C_dv12 + 
                rho * sigma_v * V1_pde_1d * S1_pde_1d * d2C_ds1v1 - r * pred1 + jump_integral_1d)
    l_p1 = torch.mean(pde_res1**2)
    l_t1 = nn.MSELoss()(pinn_1d(S_term_1d, V_term_1d, t_term_1d), C_term_1d)
    
    (l_d1 + 0.02 * l_p1 + l_t1).backward()
    opt_pinn1d.step()

    opt_nn2d.zero_grad()
    l_nn2 = nn.MSELoss()(nn_2d(S1_2d_t, S2_2d_t, V1_2d_t, V2_2d_t, t_2d_t), C_2d_t)
    l_nn2.backward()
    opt_nn2d.step()
    
    opt_pinn2d.zero_grad()
    l_d2 = nn.MSELoss()(pinn_2d(S1_2d_t, S2_2d_t, V1_2d_t, V2_2d_t, t_2d_t), C_2d_t)
    
    pred2 = pinn_2d(S1_pde_2d, S2_pde_2d, V1_pde_2d, V2_pde_2d, t_pde_2d)
    dc_ds1 = torch.autograd.grad(pred2.sum(), S1_pde_2d, create_graph=True)[0]
    dc_ds2 = torch.autograd.grad(pred2.sum(), S2_pde_2d, create_graph=True)[0]
    dc_dv1 = torch.autograd.grad(pred2.sum(), V1_pde_2d, create_graph=True)[0]
    dc_dv2 = torch.autograd.grad(pred2.sum(), V2_pde_2d, create_graph=True)[0]
    dc_dt2 = torch.autograd.grad(pred2.sum(), t_pde_2d, create_graph=True)[0]
    
    d2_ds12 = torch.autograd.grad(dc_ds1.sum(), S1_pde_2d, create_graph=True)[0]
    d2_ds22 = torch.autograd.grad(dc_ds2.sum(), S2_pde_2d, create_graph=True)[0]
    d2_dv12 = torch.autograd.grad(dc_dv1.sum(), V1_pde_2d, create_graph=True)[0]
    d2_dv22 = torch.autograd.grad(dc_dv2.sum(), V2_pde_2d, create_graph=True)[0]
    
    d2_ds1v1 = torch.autograd.grad(dc_ds1.sum(), V1_pde_2d, create_graph=True)[0]
    d2_ds2v2 = torch.autograd.grad(dc_ds2.sum(), V2_pde_2d, create_graph=True)[0]
    d2_ds1s2 = torch.autograd.grad(dc_ds1.sum(), S2_pde_2d, create_graph=True)[0]
    d2_dv1v2 = torch.autograd.grad(dc_dv1.sum(), V2_pde_2d, create_graph=True)[0]
    
    jump_integral_2d = (lam * kappa_j * S1_pde_2d * dc_ds1 + lam * kappa_j * S2_pde_2d * dc_ds2 + 
                        0.5 * lam * (mom2_y - 2 * (kappa_j + 1) + 1) * (S1_pde_2d**2) * d2_ds12 + 
                        0.5 * lam * (mom2_y - 2 * (kappa_j + 1) + 1) * (S2_pde_2d**2) * d2_ds22 + 
                        lam * (mom2_y - 2 * (kappa_j + 1) + 1) * S1_pde_2d * S2_pde_2d * d2_ds1s2)
    
    pde_res2 = (dc_dt2 + (r - lam * kappa_j) * S1_pde_2d * dc_ds1 + (r - lam * kappa_j) * S2_pde_2d * dc_ds2 + 
                kappa * (theta - V1_pde_2d) * dc_dv1 + kappa * (theta - V2_pde_2d) * dc_dv2 + 
                0.5 * V1_pde_2d * (S1_pde_2d**2) * d2_ds12 + 0.5 * V2_pde_2d * (S2_pde_2d**2) * d2_ds22 + 
                0.5 * (sigma_v**2) * V1_pde_2d * d2_dv12 + 0.5 * (sigma_v**2) * V2_pde_2d * d2_dv22 + 
                rho * sigma_v * V1_pde_2d * S1_pde_2d * d2_ds1v1 + rho * sigma_v * V2_pde_2d * S2_pde_2d * d2_ds2v2 + 
                rho_assets * torch.sqrt(V1_pde_2d * V2_pde_2d) * S1_pde_2d * S2_pde_2d * d2_ds1s2 + 
                rho_vars * (sigma_v**2) * torch.sqrt(V1_pde_2d * V2_pde_2d) * d2_dv1v2 - r * pred2 + jump_integral_2d)
    l_p2 = torch.mean(pde_res2**2)
    l_t2 = nn.MSELoss()(pinn_2d(S1_term_2d, S2_term_2d, V1_term_2d, V2_term_2d, t_term_2d), C_term_2d)
    
    (l_d2 + 0.01 * l_p2 + l_t2).backward()
    opt_pinn2d.step()

# =============================================================================
# PERFORMANCE METRICS EVALUATION
# =============================================================================
S_test = np.linspace(S_extrap_min, S_extrap_max, 30)
V_test_fixed = np.ones_like(S_test) * theta 
T_test_fixed = np.ones_like(S_test) * T
t_test_fixed_t = torch.zeros((len(S_test), 1), dtype=torch.float32)

S_test_t = torch.tensor(S_test.reshape(-1, 1), dtype=torch.float32)
V_test_t = torch.tensor(V_test_fixed.reshape(-1, 1), dtype=torch.float32)

t0 = time.time(); pr_fourier_1d = bates_call_analytical(S_test, K, V_test_fixed, T_test_fixed, r, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j); time_fourier_1d = time.time() - t0
t0 = time.time(); pr_mc_1d = monte_carlo_bates_1d(S_test, V_test_fixed, K, T_test_fixed, r, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j, M=15000); time_mc_1d = time.time() - t0

with torch.no_grad():
    t0 = time.time(); pr_nn_1d = nn_1d(S_test_t, V_test_t, t_test_fixed_t).numpy().flatten(); time_nn_1d = time.time() - t0
    t0 = time.time(); pr_pinn_1d = pinn_1d(S_test_t, V_test_t, t_test_fixed_t).numpy().flatten(); time_pinn_1d = time.time() - t0

t0 = time.time(); pr_mc_2d = monte_carlo_bates_basket_2d(S_test, S_test, V_test_fixed, V_test_fixed, K, T_test_fixed, r, kappa, theta, sigma_v, rho, rho_assets, rho_vars, lam, mu_j, sigma_j, M=15000); time_mc_2d = time.time() - t0

with torch.no_grad():
    t0 = time.time(); pr_nn_2d = nn_2d(S_test_t, S_test_t, V_test_t, V_test_t, t_test_fixed_t).numpy().flatten(); time_nn_2d = time.time() - t0
    t0 = time.time(); pr_pinn_2d = pinn_2d(S_test_t, S_test_t, V_test_t, V_test_t, t_test_fixed_t).numpy().flatten(); time_pinn_2d = time.time() - t0

print("\n" + "="*70)
print("1D SINGLE-ASSET BATES OPTION EVALUATIONS")
df_1d = pd.DataFrame({'Spot_S': S_test, 'Analytical_Fourier': pr_fourier_1d, 'Monte_Carlo_1D': pr_mc_1d, 'Standard_NN_1D': pr_nn_1d, 'PINN_1D': pr_pinn_1d})
print(df_1d.iloc[[0, 7, 15, 22, -1]].to_string(index=False))

print("\n" + "="*70)
print("2D MULTI-ASSET BATES BASKET (MAX PAYOFF) EVALUATIONS")
df_2d = pd.DataFrame({'S1_and_S2': S_test, 'Monte_Carlo_Basket': pr_mc_2d, 'Standard_NN_2D': pr_nn_2d, 'PINN_2D': pr_pinn_2d})
print(df_2d.iloc[[0, 7, 15, 22, -1]].to_string(index=False))

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

axs[0, 0].plot(S_test, pr_fourier_1d, 'k-', lw=2.5, label='Analytical Fourier (Carr-Madan)')
axs[0, 0].plot(S_test, pr_mc_1d, 'b--', lw=1.5, label='Monte Carlo Reference')
axs[0, 0].plot(S_test, pr_nn_1d, 'r:', lw=2.5, label='Standard NN (No Physics)')
axs[0, 0].plot(S_test, pr_pinn_1d, 'g-.', lw=2.5, label='Physics-Informed PINN')
axs[0, 0].axvspan(S_train_min, S_train_max, color='yellow', alpha=0.12, label="Training Boundary Strip")
axs[0, 0].set_title("1D Bates Single-Asset Option Extrapolation")
axs[0, 0].set_xlabel("Asset Spot Price (S)"); axs[0, 0].set_ylabel("Option Price")
axs[0, 0].legend(); axs[0, 0].grid(True)

axs[0, 1].plot(S_test, pr_mc_2d, 'b-', lw=2.5, label='Multi-Asset Monte Carlo')
axs[0, 1].plot(S_test, pr_nn_2d, 'r:', lw=2.5, label='Standard NN (Diverging Bounds)')
axs[0, 1].plot(S_test, pr_pinn_2d, 'g-.', lw=2.5, label='Multi-Asset PINN Framework')
axs[0, 1].axvspan(S_train_min, S_train_max, color='yellow', alpha=0.12, label="Training Boundary Strip")
axs[0, 1].set_title("2D Bates Basket Option Extrapolation (Max-Payoff)")
axs[0, 1].set_xlabel("Asset Spot Prices (S1 = S2)"); axs[0, 1].set_ylabel("Basket Option Price")
axs[0, 1].legend(); axs[0, 1].grid(True)

axs[1, 0].bar(['Fourier', 'Monte Carlo', 'Standard NN', 'PINN 1D'], [time_fourier_1d, time_mc_1d, time_nn_1d, time_pinn_1d], color=['black','blue','red','green'])
axs[1, 0].set_yscale('log')
axs[1, 0].set_title("1D Computational Inference Execution Speeds (Seconds)")
axs[1, 0].set_ylabel("Log Scale Time Duration")
axs[1, 0].grid(True, which="both", ls="--", alpha=0.5)

axs[1, 1].bar(['Monte Carlo 2D', 'Standard NN 2D', 'PINN 2D Basket'], [time_mc_2d, time_nn_2d, time_pinn_2d], color=['blue','red','green'])
axs[1, 1].set_yscale('log')
axs[1, 1].set_title("2D Basket Computational Inference Execution Speeds (Seconds)")
axs[1, 1].set_ylabel("Log Scale Time Duration")
axs[1, 1].grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.show()
