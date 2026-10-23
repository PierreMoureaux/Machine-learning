import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for clinical reproducibility
np.random.seed(42)
torch.manual_seed(42)

# =============================================================================
# HULL-WHITE 1D MODEL PARAMETERS
# =============================================================================
a = 0.1  # Mean reversion speed
sigma = 0.03  # Volatility of short rate
theta_const = 0.05  # Baseline drift parameter (assumed constant for structure analysis)
T = 1.0  # Maturity of the bond

# Splitting bounds for generalization tests
r_train_min, r_train_max = 0.01, 0.07
r_extrap_min, r_extrap_max = -0.05, 0.15


# =============================================================================
# 1. ANALYTICAL CLOSED-FORM SOLUTION
# =============================================================================
def hull_white_analytical_bond(r_vec, t_vec, a, sigma, theta, T):
    N = len(r_vec)
    prices = np.zeros(N)
    for i in range(N):
        rt, t = r_vec[i], t_vec[i]
        tau = T - t
        if tau <= 1e-5:
            prices[i] = 1.0
            continue
        B = (1.0 - np.exp(-a * tau)) / a
        A_term = (theta - (sigma ** 2) / (2 * a ** 2)) * (B - tau) - (sigma ** 2 * B ** 2) / (4 * a)
        A = np.exp(A_term)
        prices[i] = A * np.exp(-B * rt)
    return prices


# =============================================================================
# 2. MONTE CARLO SIMULATOR (EXACT CONDITIONAL EULER SCHEMA)
# =============================================================================
def monte_carlo_hull_white(r_vec, t_vec, a, sigma, theta, T, M=15000, N_steps=80):
    N_inputs = len(r_vec)
    prices = np.zeros(N_inputs)
    for i in range(N_inputs):
        r0, t0 = r_vec[i], t_vec[i]
        tau = T - t0
        if tau <= 1e-5:
            prices[i] = 1.0
            continue
        dt = tau / N_steps
        r_paths = np.ones(M) * r0
        integral_r = np.zeros(M)
        for _ in range(N_steps):
            Z = np.random.standard_normal(M)
            r_paths_next = r_paths + (theta - a * r_paths) * dt + sigma * np.sqrt(dt) * Z
            integral_r += 0.5 * (r_paths + r_paths_next) * dt
            r_paths = r_paths_next
        prices[i] = np.mean(np.exp(-integral_r))
    return prices


# =============================================================================
# 3. FINITE DIFFERENCE METHOD - GRID SOLVER (IMPLICIT BACKWARD SCHEME)
# =============================================================================
def finite_difference_hull_white(r_vec, t_target, a, sigma, theta, T):
    r_max_grid = 0.40
    r_min_grid = -0.20
    N, NJ = 200, 200
    dr = (r_max_grid - r_min_grid) / N
    dt = T / NJ
    r_grid = np.linspace(r_min_grid, r_max_grid, N + 1)
    grid = np.ones(N + 1)
    alpha_coeffs = np.zeros(N + 1)
    beta_coeffs = np.zeros(N + 1)
    gamma_coeffs = np.zeros(N + 1)
    for j in range(1, N):
        curr_r = r_grid[j]
        drift = theta - a * curr_r
        alpha_coeffs[j] = 0.5 * dt * (drift / dr - (sigma ** 2) / (dr ** 2))
        beta_coeffs[j] = 1.0 + dt * ((sigma ** 2) / (dr ** 2) + curr_r)
        gamma_coeffs[j] = -0.5 * dt * (drift / dr + (sigma ** 2) / (dr ** 2))
    A = np.zeros((N + 1, N + 1))
    A[0, 0], A[N, N] = 1.0, 1.0
    for j in range(1, N):
        A[j, j - 1], A[j, j], A[j, j + 1] = alpha_coeffs[j], beta_coeffs[j], gamma_coeffs[j]
    A_inv = np.linalg.inv(A)
    for _ in range(NJ):
        rhs = np.copy(grid)
        rhs[0] = 1.2
        rhs[N] = 0.0
        grid = A_inv.dot(rhs)
    return np.interp(r_vec, r_grid, grid)


# =============================================================================
# 4. DEEP NEURAL NETWORK MODEL STRUCTURE
# =============================================================================
class HWNet(nn.Module):
    def __init__(self):
        super(HWNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, r_tensor, t_tensor):
        inputs = torch.cat([r_tensor, t_tensor / T], dim=1)
        out = self.net(inputs)
        return torch.nn.functional.softplus(out) + 0.1


# =============================================================================
# DATASET COMPILATION AND TENSOR SETUP
# =============================================================================
num_train = 1000
r_train_data = np.random.uniform(r_train_min, r_train_max, (num_train, 1))
t_train_data = np.random.uniform(0.0, T, (num_train, 1))

P_train_targets = hull_white_analytical_bond(r_train_data[:, 0], t_train_data[:, 0], a, sigma, theta_const, T).reshape(
    -1, 1)

r_train_t = torch.tensor(r_train_data, dtype=torch.float32)
t_train_t = torch.tensor(t_train_data, dtype=torch.float32)
P_train_t = torch.tensor(P_train_targets, dtype=torch.float32)

ann_model = HWNet()
pinn_model = HWNet()

optimizer_ann = optim.Adam(ann_model.parameters(), lr=0.005)
optimizer_pinn = optim.Adam(pinn_model.parameters(), lr=0.003)

num_pde = 2000
r_pde = torch.tensor(np.random.uniform(r_extrap_min, r_extrap_max, (num_pde, 1)), dtype=torch.float32,
                     requires_grad=True)
t_pde = torch.tensor(np.random.uniform(0.0, T, (num_pde, 1)), dtype=torch.float32, requires_grad=True)

r_term = torch.tensor(np.random.uniform(r_extrap_min, r_extrap_max, (600, 1)), dtype=torch.float32)
t_term = torch.ones_like(r_term) * T
P_term_targets = torch.ones_like(r_term)

# =============================================================================
# OPTIMIZATION CYCLES ENFORCING OPERATOR EQUATIONS
# =============================================================================
print("Training Deep Learning models for Hull-White framework...")

epochs = 1200
for epoch in range(epochs):
    optimizer_ann.zero_grad()
    loss_ann = nn.MSELoss()(ann_model(r_train_t, t_train_t), P_train_t)
    loss_ann.backward()
    optimizer_ann.step()

    optimizer_pinn.zero_grad()
    loss_data_pinn = nn.MSELoss()(pinn_model(r_train_t, t_train_t), P_train_t)

    pred_pde = pinn_model(r_pde, t_pde)
    dP_dr = torch.autograd.grad(pred_pde.sum(), r_pde, create_graph=True)[0]
    dP_dt = torch.autograd.grad(pred_pde.sum(), t_pde, create_graph=True)[0]
    d2P_dr2 = torch.autograd.grad(dP_dr.sum(), r_pde, create_graph=True)[0]

    pde_residual = dP_dt + (theta_const - a * r_pde) * dP_dr + 0.5 * (sigma ** 2) * d2P_dr2 - r_pde * pred_pde
    loss_pde = torch.mean(pde_residual ** 2)

    loss_terminal = nn.MSELoss()(pinn_model(r_term, t_term), P_term_targets)

    total_pinn_loss = loss_data_pinn + 0.1 * loss_pde + loss_terminal
    total_pinn_loss.backward()
    optimizer_pinn.step()

# =============================================================================
# METRIC EVALUATION AND PERFORMANCE INFERENCE PROFILING
# =============================================================================
r_test = np.linspace(r_extrap_min, r_extrap_max, 40)
t_test_fixed = np.zeros_like(r_test)
t_test_fixed_t = torch.zeros((len(r_test), 1), dtype=torch.float32)
r_test_t = torch.tensor(r_test.reshape(-1, 1), dtype=torch.float32)

t0 = time.time();
pr_analytical = hull_white_analytical_bond(r_test, t_test_fixed, a, sigma, theta_const, T);
time_an = time.time() - t0
t0 = time.time();
pr_monte_carlo = monte_carlo_hull_white(r_test, t_test_fixed, a, sigma, theta_const, T, M=30000);
time_mc = time.time() - t0
t0 = time.time();
pr_finite_diff = finite_difference_hull_white(r_test, 0.0, a, sigma, theta_const, T);
time_fd = time.time() - t0

with torch.no_grad():
    t0 = time.time();
    pr_ann = ann_model(r_test_t, t_test_fixed_t).numpy().flatten();
    time_ann_inf = time.time() - t0
    t0 = time.time();
    pr_pinn = pinn_model(r_test_t, t_test_fixed_t).numpy().flatten();
    time_pinn_inf = time.time() - t0

# =============================================================================
# PRINT RESULTS TO CONSOLE AND TRACE GRAPHS
# =============================================================================
print("\n" + "=" * 78)
print("1D HULL-WHITE ZERO-COUPON BOND PRICING RESULTS (t=0)")
df_results = pd.DataFrame({
    'Short_Rate_r': r_test,
    'Exact_Analytical': pr_analytical,
    'Implicit_FD_Grid': pr_finite_diff,
    'Monte_Carlo': pr_monte_carlo,
    'Standard_ANN': pr_ann,
    'Physics_PINN': pr_pinn
})
print(df_results.iloc[[0, 10, 20, 30, -1]].to_string(index=False))

fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Panel 1: Price Comparisons & Extrapolation Domain
axs[0].plot(r_test, pr_analytical, 'k-', lw=2.5, label='Exact Analytical Form')
axs[0].plot(r_test, pr_finite_diff, 'o', markersize=4, label='Implicit FD Grid Solver', alpha=0.7)
axs[0].plot(r_test, pr_monte_carlo, 'b--', lw=1.5, label='Monte Carlo Simulation')
axs[0].plot(r_test, pr_ann, 'r:', lw=2.5, label='Standard ANN (Data Only)')
axs[0].plot(r_test, pr_pinn, 'g-.', lw=2.5, label='Physics-Informed PINN')
axs[0].axvspan(r_train_min, r_train_max, color='yellow', alpha=0.12, label="Training Sample Corridor")
axs[0].set_title("Hull-White Bond Valuation & Out-of-Sample Extrapolation")
axs[0].set_xlabel("Short Interest Rate (r)");
axs[0].set_ylabel("Bond Value P(r, 0)")
axs[0].legend();
axs[0].grid(True)

# Panel 2: Execution Computational Speeds Profile
axs[1].bar(['Analytical', 'Implicit FD', 'Monte Carlo', 'Standard NN', 'PINN Infe.'],
           [time_an, time_fd, time_mc, time_ann_inf, time_pinn_inf],
           color=['black', 'orange', 'blue', 'red', 'green'])
axs[1].set_yscale('log')
axs[1].set_title("Computational Inference Execution Speeds Profile")
axs[1].set_ylabel("Log Scale Execution Latency (Seconds)")
axs[1].grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.show()
