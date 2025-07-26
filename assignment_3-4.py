import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Experimental data
t_data = np.array([0, 1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36, 38, 41])
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
y0 = [T0, 0, 0, 100]  # [T, E, I, V]

# Model
def base_model(Y, t, λ, β, k, δ, p, c):
    T, E, I, V = Y
    dTdt = λ * T - β * T * V
    dEdt = β * T * V - k * E
    dIdt = k * E - δ * I
    dVdt = p * I - c * V
    return [dTdt, dEdt, dIdt, dVdt]

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

# SSR function (only uses T)
def ssr_segment_0_41(params):
    λ, β, k, δ, p, c = params
    total_ssr = 0

    # Define time masks for each segment
    t_mask_0_3 = (t_data >= 0) & (t_data <= 3)
    t_mask_3_6 = (t_data >= 3) & (t_data <= 6)
    t_mask_6_41 = (t_data >= 6) & (t_data <= 41)

    t_seg_0_3 = t_data[t_mask_0_3]
    t_seg_3_6 = t_data[t_mask_3_6]
    t_seg_6_41 = t_data[t_mask_6_41]

    for T_data in T_data_list:
        try:
            # --- Segment 1: (0–3) ---
            T_0_3 = T_data[t_mask_0_3]
            mask_0_3 = ~np.isnan(T_0_3)
            y0_local = [T0, 0, 0, 100]
            sol_0_3 = odeint(base_model, y0_local, t_seg_0_3[mask_0_3], args=(λ, β, k, δ, p, c))
            T_model_0_3 = sol_0_3[:, 0]
            total_ssr += np.sum((T_0_3[mask_0_3] - T_model_0_3) ** 2)

        except Exception as e:
            print(f"Error during fitting for a replicate: {e}")
            return np.inf

    for T_data in T_data_list:
            try:
                # --- Segment 2: (3–6) ---
                T_3_6 = T_data[t_mask_3_6]
                mask_3_6 = ~np.isnan(T_3_6)
                y0_3_6 = sol_0_3[-1].copy()
                y0_3_6[3] += 100  # Inject virus
                sol_3_6 = odeint(base_model, y0_3_6, t_seg_3_6[mask_3_6], args=(λ, β, k, δ, p, c))
                T_model_3_6 = sol_3_6[:, 0]
                total_ssr += np.sum((T_3_6[mask_3_6] - T_model_3_6) ** 2)
            except Exception as e:
                print(f"Error during fitting for a replicate: {e}")
                return np.inf

    for T_data in T_data_list:
        try:
            # --- Segment 3: (6–41) ---
            T_6_41 = T_data[t_mask_6_41]
            mask_6_41 = ~np.isnan(T_6_41)
            y0_6_41 = sol_3_6[-1].copy()
            y0_6_41[3] += 100  # Inject virus again
            sol_6_41 = odeint(base_model, y0_6_41, t_seg_6_41[mask_6_41], args=(λ, β, k, δ, p, c))
            T_model_6_41 = sol_6_41[:, 0]
            total_ssr += np.sum((T_6_41[mask_6_41] - T_model_6_41) ** 2)
        except Exception as e:
            print(f"Error during fitting for a replicate: {e}")
            return np.inf

    return total_ssr




#def ssr_segment_0_41(params):
   # return ssr_segment_0_3(params) + ssr_segment_3_6(params) + ssr_segment_6_41(params)


# Initial guess and bounds:  λ, β, k, δ, p, c
initial_guess = [0.1, 1e-8, 0.1, 0.1, 1e5, 0.1]
bounds = [
    (0.00001, 1),      # λ
    (1e-10, 1e-2),    # β
    (0.0001, 1),      # k
    (0.01, 2),      # δ
    (1e3, 1e7),       # p
    (0.01, 2)       # c
]

# Fit parameters
result = minimize(ssr_segment_0_41, initial_guess,
                  bounds=bounds, method='L-BFGS-B')


λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit = result.x
print(f"Fitted λ: {λ_fit:.4f}")
print(f"Fitted β: {β_fit:.4f}")
print(f"Fitted k: {k_fit:.4f}")
print(f"Fitted δ: {δ_fit:.4f}")
print(f"Fitted p: {p_fit:.4f}")
print(f"Fitted c: {c_fit:.4f}")

T_pred_full = []
t_pred_full = []

# --- Segment 1: 0–3 ---
t_seg_0_3 = t_data[(t_data >= 0) & (t_data <= 3)]
y0_0_3 = [T0, 0, 0, 100]
sol_0_3 = odeint(base_model, y0_0_3, t_seg_0_3, args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))
T_pred_full.extend(sol_0_3[:, 0])
t_pred_full.extend(t_seg_0_3)
y0_3_6 = sol_0_3[-1].copy()
y0_3_6[3] += 100  # Add virus

# --- Segment 2: 3–6 ---
t_seg_3_6 = t_data[(t_data > 3) & (t_data <= 6)]  # exclude 3 to avoid duplication
sol_3_6 = odeint(base_model, y0_3_6, t_seg_3_6, args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))
T_pred_full.extend(sol_3_6[:, 0])
t_pred_full.extend(t_seg_3_6)
y0_6_41 = sol_3_6[-1].copy()
y0_6_41[3] += 100  # Add virus again

# --- Segment 3: 6–41 ---
t_seg_6_41 = t_data[(t_data > 6)]  # exclude 6 to avoid duplication
sol_6_41 = odeint(base_model, y0_6_41, t_seg_6_41, args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))
T_pred_full.extend(sol_6_41[:, 0])
t_pred_full.extend(t_seg_6_41)

T_pred = np.array(T_pred_full)
t_pred = np.array(t_pred_full)

# Predict using best-fit parameters
#PBSpred = odeint(base_model, y0, t_data, args=tuple(result.x))
#T_pred = PBSpred[:, 0]

# Plot
for T_data in T_data_list:
    mask = ~np.isnan(T_data)
    plt.scatter(t_data[mask], T_data[mask], s=8, color='orangered', label="Data" if T_data is T_data_list[0] else "")

plt.plot(t_pred, T_pred, color='blue', label='Best Fit')
plt.title("Tumor Growth (Model Fit)")
plt.ylabel('Tumor volume(mm³)')
plt.xlabel('Time (days)')
plt.legend()
plt.grid()
plt.show()