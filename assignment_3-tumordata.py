import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

t_data = np.array([0, 1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36, 38, 41])

# tumor data
T_data_list = [
    np.array([102, 107, 116, 125, 137, 149, 167, 177, 187, 291, 367, 386, 441, 460, 605, 733, 846, 922, np.nan]),
    np.array([125, 135, 143, 156, 169, 201, 222, 238, 251, 336, 469, 500, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
    np.array([110, 119, 124, 133, 144, 188, 208, 223, 233, 354, 451, 477, 562, 587, 679, np.nan, np.nan, np.nan, np.nan]),
    np.array([128, 135, 143, 159, 173, 211, 237, 256, 272, 405, 471, 492, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
    np.array([139, 148, 157, 170, 187, 206, 223, 237, 250, 321, 382, 411, 425, 443, 518, 570, 617, 661, 763])
]

T0 = 1.76
y0 = [T0, 0, 0, 100]  # [T, E, I, V]

def base_model(Y, t, λ, β, k, δ, p, c):
    T, E, I, V = Y
    dTdt = λ * T - β * T * V
    dEdt = β * T * V - k * E
    dIdt = k * E - δ * I
    dVdt = p * I - c * V
    return [dTdt, dEdt, dIdt, dVdt]

# SSR using all segments
def ssr_segment(params):
    λ, β, k, δ, p, c = params
    total_ssr = 0

    for T_data in T_data_list:
        try:
            mask = ~np.isnan(T_data)
            t_clean = t_data[mask]
            T_clean = T_data[mask]

            y_current = y0.copy()
            t_pred = []
            T_pred = []

            segment_bounds = [(0, 25), (25, 28), (28, 31), (31, 47)]

            for t_start, t_end in segment_bounds:
                t_seg = t_clean[(t_clean >= t_start) & (t_clean < t_end)]
                if len(t_seg) == 0:
                    continue
                sol = odeint(base_model, y_current, t_seg, args=(λ, β, k, δ, p, c))
                T_model = sol[:, 0]

                T_actual = T_clean[(t_clean >= t_start) & (t_clean < t_end)]

                total_ssr += np.sum((T_actual - T_model) ** 2)
                y_current = sol[-1].copy()
                y_current[3] += 100  # virus boost each segment

        except Exception as e:
            print(f"Error: {e}")
            return np.inf

    return total_ssr

# Initial guess and bounds:  λ, β, k, δ, p, c
initial_guess = [0.1, 1e-8, 0.1, 0.1, 1e5, 0.1]
bounds = [
    (0.00001, 1),      # λ
    (1e-10, 1e-2),     # β
    (0.0001, 1),       # k
    (0.01, 2),         # δ
    (1e3, 1e7),        # p
    (0.01, 2)          # c
]

# Fit parameters
result = minimize(ssr_segment, initial_guess, bounds=bounds, method='L-BFGS-B')
λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit = result.x

print(f"Fitted λ: {λ_fit:.4f}")
print(f"Fitted β: {β_fit:.4f}")
print(f"Fitted k: {k_fit:.4f}")
print(f"Fitted δ: {δ_fit:.4f}")
print(f"Fitted p: {p_fit:.4f}")
print(f"Fitted c: {c_fit:.4f}")

# Full prediction using fitted parameters
y_current = y0.copy()
t_pred_full = []
T_pred_full = []
segment_bounds = [(0, 25), (25, 28), (28, 31), (31, 47)]

for t_start, t_end in segment_bounds:
    t_seg = t_data[(t_data >= t_start) & (t_data < t_end)]
    if len(t_seg) == 0:
        continue
    sol = odeint(base_model, y_current, t_seg, args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))
    T_pred_full.extend(sol[:, 0])
    t_pred_full.extend(t_seg)
    y_current = sol[-1].copy()
    y_current[3] += 100  # virus injection

# Final arrays
T_pred = np.array(T_pred_full)
t_pred = np.array(t_pred_full)

# Plot
for T_data in T_data_list:
    mask = ~np.isnan(T_data)
    plt.scatter(t_data[mask], T_data[mask], s=8, color='orangered', label="Data" if T_data is T_data_list[0] else "")

plt.plot(t_pred, T_pred, color='blue', label='Best Fit')
plt.title("Tumor Growth (Model Fit)")
plt.ylabel('Tumor volume (mm³)')
plt.xlabel('Time (days)')
plt.legend()
plt.grid()
plt.show()
