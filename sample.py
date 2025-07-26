import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Data
v_data = np.array([0.04167, 1, 3, 7, 28])
V_data_list = [
   np.array([1.372643, 0.018196, 1.158547, 99.04117, 26509.89]),
   np.array([0.01, 0.012721, 0.01, 585.1756, 6010.401]),
   np.array([np.nan, 0.01, 0.01, 0.275531, 92476.8])
]

T0 = 1.76
y0 = [T0, 0, 1, 100]  # [T, E, I, V]

# Model
def base_model(Y, t, λ, β, k, δ, p, c):
    T, E, I, V = Y
    dTdt = λ * T - β * T * V
    dEdt = β * T * V - k * E
    dIdt = k * E - δ * I
    dVdt = p * I - c * V
    return [dTdt, dEdt, dIdt, dVdt]

# Log-scale SSR
def ssr_segment_0_28(params):
    λ, β, k, δ, p, c = params
    total_ssr = 0
    v_mask_0_1 = (v_data >= 0.0147) & (v_data <= 1)
    v_mask_1_7 = (v_data >= 1) & (v_data <= 7)
    v_mask_7_28 = (v_data >= 7) & (v_data <= 28)

    v_seg_0_1 = v_data[v_mask_0_1]
    v_seg_1_7 = v_data[v_mask_1_7]
    v_seg_7_28 = v_data[v_mask_7_28]

    for V_data in V_data_list:
        try:
            # Segment 1: 0–1
            V_0_1 = V_data[v_mask_0_1]
            mask_0_1 = ~np.isnan(V_0_1)
            y0_local = [T0, 0, 1, 100]
            sol_0_1 = odeint(base_model, y0_local, v_seg_0_1[mask_0_1], args=(λ, β, k, δ, p, c))
            V_model = np.clip(sol_0_1[:, 3], 1e-10, None)
            V_obs = np.clip(V_0_1[mask_0_1], 1e-10, None)
            total_ssr += np.sum((np.log10(V_obs) - np.log10(V_model)) ** 2)

        except Exception as e:
            print(f"Error during Segment 0–1: {e}")
            return np.inf

    for V_data in V_data_list:
        try:
            # Segment 2: 1–7
            V_1_7 = V_data[v_mask_1_7]
            mask_1_7 = ~np.isnan(V_1_7)
            y0_1_7 = sol_0_1[-1].copy()
            y0_1_7[3] += 100
            sol_1_7 = odeint(base_model, y0_1_7, v_seg_1_7[mask_1_7], args=(λ, β, k, δ, p, c))
            V_model = np.clip(sol_1_7[:, 3], 1e-10, None)
            V_obs = np.clip(V_1_7[mask_1_7], 1e-10, None)
            total_ssr += np.sum((np.log10(V_obs) - np.log10(V_model)) ** 2)

        except Exception as e:
            print(f"Error during Segment 1–7: {e}")
            return np.inf

    for V_data in V_data_list:
        try:
            # Segment 3: 7–28
            V_7_28 = V_data[v_mask_7_28]
            mask_7_28 = ~np.isnan(V_7_28)
            y0_7_28 = sol_1_7[-1].copy()
            y0_7_28[3] += 100
            sol_7_28 = odeint(base_model, y0_7_28, v_seg_7_28[mask_7_28], args=(λ, β, k, δ, p, c))
            V_model = np.clip(sol_7_28[:, 3], 1e-10, None)
            V_obs = np.clip(V_7_28[mask_7_28], 1e-10, None)
            total_ssr += np.sum((np.log10(V_obs) - np.log10(V_model)) ** 2)

        except Exception as e:
            print(f"Error during Segment 7–28: {e}")
            return np.inf

    return total_ssr

# Optimization
initial_guess = [0.1, 1e-5, 0.1, 0.1, 1e6, 0.5]
bounds = [
    (0.00001, 1),    # λ
    (1e-10, 1e-2),   # β
    (0.0001, 1),     # k
    (0.01, 2),       # δ
    (1e3, 1e10),     # p
    (0.01, 2)        # c
]

result = minimize(ssr_segment_0_28, initial_guess, bounds=bounds, method='L-BFGS-B')

λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit = result.x
print(f"Fitted λ: {λ_fit:.4f}")
print(f"Fitted β: {β_fit:.4f}")
print(f"Fitted k: {k_fit:.4f}")
print(f"Fitted δ: {δ_fit:.4f}")
print(f"Fitted p: {p_fit:.4f}")
print(f"Fitted c: {c_fit:.4f}")

# Predict values using fitted params
V_pred_full = []
v_pred_full = []

v_seg_0_1 = v_data[(v_data >= 0.04167) & (v_data <= 1)]
y0_0_1 = [T0, 0, 0, 100]
sol_0_1 = odeint(base_model, y0_0_1, v_seg_0_1, args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))
V_pred_full.extend(sol_0_1[:, 3])
v_pred_full.extend(v_seg_0_1)
y0_1_7 = sol_0_1[-1].copy()
y0_0_1[3] += 100

v_seg_1_7 = v_data[(v_data > 1) & (v_data <= 7)]
sol_1_7 = odeint(base_model, y0_1_7, v_seg_1_7, args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))
V_pred_full.extend(sol_1_7[:, 3])
v_pred_full.extend(v_seg_1_7)
y0_7_28 = sol_1_7[-1].copy()
y0_1_7[3] += 100

v_seg_7_28 = v_data[(v_data > 7) & (v_data <= 28)]
sol_7_28 = odeint(base_model, y0_7_28, v_seg_7_28, args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))
V_pred_full.extend(sol_7_28[:, 3])
v_pred_full.extend(v_seg_7_28)

V_pred = np.array(V_pred_full)
v_pred = np.array(v_pred_full)

# Plot (Log Y)
for V_data in V_data_list:
    mask = ~np.isnan(V_data)
    plt.scatter(v_data[mask], V_data[mask], s=8, color='orangered', label="Data" if V_data is V_data_list[0] else "")

plt.plot(v_pred, V_pred, color='blue', label='Best Fit')
plt.title("Virus Growth (Model Fit - Log SSR)")
plt.ylabel('Virus volume (mm³)')
plt.xlabel('Time (days)')
plt.yscale('log')  # Log scale for y-axis
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.show()
