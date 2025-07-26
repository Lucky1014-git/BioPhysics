import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Data
PBSx = np.array([25, 26, 28, 31, 33, 35, 38, 40, 42, 45, 47, 49, 52, 54, 56, 59])
PBSy = np.array([111, 120, 131, 141, 154, 172, 185, 196, 215, 359, 521, 546, 708, 759, 853, 1021])

# Logistic Model
def basemodel(T, t, lam, K):
    return lam * T * (1 - (T / K))

# Sum of Squares Residual (SSR) for fitting
def PBSSSR(params):
    lam, K = params
    # Integrate starting from T0 = 1.76 over the range of actual data times
    pred_values = odeint(basemodel, 1.76, PBSx - PBSx[0], args=(lam, K)).flatten()
    SSR = np.sum((pred_values - PBSy)**2)
    return SSR

# Bounds for lambda and K
bounds = [(0.001, 1), (PBSy.max(), 10000)]
PBSfit = minimize(PBSSSR, [0.1, 1500], method='L-BFGS-B', bounds=bounds)

# Final Fitted Parameters
lam_best, K_best = PBSfit.x
print(f"Best fit: lambda = {lam_best:.3f}, K = {K_best:.1f}")

# Time range starting from 0
t_values = np.arange(0, PBSx[-1] - PBSx[0] + 1)

# Final Model Predictions (from t = 0, with T0 = 1.76)
PBSpred_full = odeint(basemodel, 1.76, t_values, args=(lam_best, K_best)).flatten()

# Extract prediction times matching actual data
aligned_predictions = PBSpred_full[PBSx - PBSx[0]]

# Plot
# Plot the full prediction starting from Day 0
plt.figure(figsize=(10, 6))
plt.plot(t_values, PBSpred_full, color='blue', label=f"Fit: lambda = {lam_best:.3f}, K = {K_best:.1f}")

# Plot the actual data (without day 0) as before
plt.scatter(PBSx, PBSy, s=30, color='orangered', label="Actual Data")

# Mark the initial point at t = 0
plt.scatter(0, 1.76, color='green', s=100, marker='o', label='Initial Point (T0 = 1.76)')

plt.title("Tumor Growth (Logistic Model) Including Day 0")
plt.ylabel('Volume (mm³)')
plt.xlabel('Time (days)')
plt.legend()
plt.show()

