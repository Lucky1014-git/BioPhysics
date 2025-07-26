import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Experimental data
t_data = np.array([0, 25, 26, 28, 31, 33, 35, 38, 40, 42, 45, 47, 49, 52, 54, 56, 59])
T_data = np.array([1.76, 111, 120, 131, 141, 154, 172, 185, 196, 215, 359, 521, 546, 708, 759, 853, 1021])

# Initial conditions
T0 = 1.76
y0 = [T0, 0, 0, 0]  # [T, E, I, V]

# Model
def base_model(Y, t, λ, β, k, δ, p, c):
    T, E, I, V = Y
    dTdt = λ * T - β * T * V
    dEdt = β * T * V - k * E
    dIdt = k * E - δ * I
    dVdt = p * I - c * V
    return [dTdt, dEdt, dIdt, dVdt]

# SSR function (only uses T)
def ssr_tumor_only(params):
    λ, β, k, δ, p, c = params
    try:
        sol = odeint(base_model, y0, t_data, args=(λ, β, k, δ, p, c))
        T_model = sol[:, 0]
        ssr = np.sum((T_data - T_model) ** 2)
        print(ssr)
        return ssr
    except:
        return np.inf

# Initial guess and bounds:  λ, β, k, δ, p, c
initial_guess = [0.1, 1e-5, 0.1, 0.1, 1e5, 0.1]
bounds = [
    (0.0001, 1),      # λ
    (1e-10, 1e-2),    # β
    (0.0001, 1),      # k
    (0.0001, 1),      # δ
    (1e3, 1e7),       # p
    (0.0001, 1)       # c
]

# Fit parameters
result = minimize(ssr_tumor_only, initial_guess,
                  bounds=bounds, method='L-BFGS-B')
λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit = result.x
print(f"Fitted λ: {λ_fit:.4f}")
print(f"Fitted β: {β_fit:.4f}")
print(f"Fitted k: {k_fit:.4f}")
print(f"Fitted δ: {δ_fit:.4f}")
print(f"Fitted p: {p_fit:.4f}")
print(f"Fitted c: {c_fit:.4f}")

# Predict using best-fit parameters
PBSpred = odeint(base_model, y0, t_data, args=tuple(result.x))
T_pred = PBSpred[:, 0]

# Plot
plt.scatter(t_data, T_data, s=8, color='orangered', label="Data")
plt.plot(t_data, T_pred, color='blue', label='Best Fit')
plt.title("Tumor Growth (Model Fit)")
plt.ylabel('Tumor volume(mm³)')
plt.xlabel('Time (days)')
plt.legend()
plt.grid()
plt.show()
