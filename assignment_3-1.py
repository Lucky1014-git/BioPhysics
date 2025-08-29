import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Experimental data
#t_data = np.array([0, 25, 26, 28, 31, 33, 35, 38, 40, 42, 45, 47, 49, 52, 54, 56, 59, 61, 63, 66])
t_data = np.array([24, 26, 28, 31, 33, 35, 38, 40, 42, 45, 47, 49, 52, 54, 56, 59, 61, 63, 66])

# Each y column (fill missing values with np.nan)
T_data_list = [
    np.array([111, 120, 131, 141, 154, 172, 185, 196, 215, 359, 521, 546, 708, 759, 853, 1021, np.nan, np.nan, np.nan]),
    np.array([115, 123, 132, 144, 154, 179, 195, 207, 219, 413, 447, 470, 599, 637, 993, np.nan, np.nan, np.nan, np.nan]),
    np.array([115, 123, 127, 136, 148, 166, 181, 195, 204, 310, 352, 371, 510, 538, 637, 759, 933, 1001, 1247]),
    np.array([148, 156, 168, 183, 202, 216, 237, 254, 269, 370, 447, 472, 661, 690, 783, 985, 1231, np.nan, np.nan]),
    np.array([102, 110, 116, 126, 138, 175, 191, 207, 222, 285, 345, 364, 638, 674, 759, np.nan, np.nan, np.nan, np.nan])
]
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
def ssr_all(params):
    λ, β, k, δ, p, c = params
    total_ssr = 0
    for T_data in T_data_list:
        mask = ~np.isnan(T_data)
        try:
            sol = odeint(base_model, y0, t_data[mask], args=(λ, β, k, δ, p, c))
            T_model = sol[:, 0]
            total_ssr += np.sum((T_data[mask] - T_model) ** 2)
        except:
            return np.inf
    return total_ssr

# Initial guess and bounds:  λ, β, k, δ, p, c
initial_guess = [0.10, 1e-5, 0.1, 0.1, 1e5, 0.1]
bounds = [
    (0.10, 0.35),      # λ: Narrower range around 0.3
    (0, 1e-2),         # β
    (0.0001, 1),       # k
    (0.0001, 1),       # δ
    (1e3, 1e7),        # p
    (0.0001, 1)        # c
]


# Fit parameters
result = minimize(ssr_all, initial_guess,
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
for T_data in T_data_list:
    mask = ~np.isnan(T_data)
    plt.scatter(t_data[mask], T_data[mask], s=8, color='orangered', label="Data" if T_data is T_data_list[0] else "")

plt.plot(t_data, T_pred, color='blue', label='Best Fit')
plt.title("Tumor Growth (Model Fit)")
plt.ylabel('Tumor volume(mm³)')
plt.xlabel('Time (days)')
plt.legend()
plt.grid()
plt.show()