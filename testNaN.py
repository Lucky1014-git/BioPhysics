import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Experimental data
t_data = np.array([24, 26, 28, 31, 33, 35, 38, 40, 42, 45, 47, 49, 52, 54, 56, 59, 61, 63, 66])
T_data_list = [
    np.array([102, 125, 110, 128, 139]),
    np.array([107, 135, 119, 135, 148]),
    np.array([116, 143, 124, 143, 157]),
    np.array([125, 156, 133, 159, 170]),
    np.array([137, 169, 144, 173, 187]),
    np.array([149, 201, 188, 211, 206]),
    np.array([167, 222, 208, 237, 223]),
    np.array([187, 251, 233, 272, 250]),
    np.array([291, 336, 354, 405, 321]),
    np.array([367, 469, 451, 471, 382]),
    np.array([386, 500, 477, 492, 411]),
    np.array([441, np.nan, 562, np.nan, 425]),
    np.array([460, np.nan, 587, np.nan, 443]),
    np.array([605, np.nan, 679, np.nan, 518]),
    np.array([733, np.nan, np.nan, np.nan, 570]),
    np.array([846, np.nan, np.nan, np.nan, 617]),
    np.array([922, np.nan, np.nan, np.nan, 661]),
    np.array([np.nan, np.nan, np.nan, np.nan, 763])
]

v_data = np.array([0.04167, 1, 3, 7, 28])
V_data_list = [
    np.array([1.372643, 0.018196, 1.158547, 99.04117, 26509.89]),
    np.array([0.01, 0.012721, 0.01, 585.1756, 6010.401]),
    np.array([np.nan, 0.01, 0.01, 0.275531, 92476.8])
]


def base_model(Y, t, λ, β, k, δ, p, c):
    T, E, I, V = Y
    dTdt = λ * T - β * T * V
    dEdt = β * T * V - k * E
    dIdt = k * E - δ * I
    dVdt = p * I - c * V
    return [dTdt, dEdt, dIdt, dVdt]


def safe_log10(x):
    """Safe log10 that returns NaN for invalid values"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x > 0, np.log10(x), np.nan)


def ssr(params):
    λ, β, k, δ, p, c = params
    y0 = [1.76, 0, 0, 0]

    # Segment 1: 0-25 days
    y1 = odeint(base_model, y0, [0, 24, 25], args=(λ, 10 ** β, k, δ, 10 ** p, c))
    y02 = y1[-1, :]
    y02[3] = y02[3] + 100

    # Segment 2: 25-28 days
    y2 = odeint(base_model, y02, [25, 25.04167, 26, 27, 28], args=(λ, 10 ** β, k, δ, 10 ** p, c))
    y03 = y2[-1, :]
    y03[3] = y03[3] + 100

    # Segment 3: 28-31 days
    y3 = odeint(base_model, y03, [28, 29, 31], args=(λ, 10 ** β, k, δ, 10 ** p, c))
    y04 = y3[-1, :]
    y04[3] = y04[3] + 100

    # Segment 4: 31-66 days
    y4 = odeint(base_model, y04, [31, 33, 35, 38, 40, 42, 45, 47, 49, 52, 54, 56, 59, 61, 63, 66],
                args=(λ, 10 ** β, k, δ, 10 ** p, c))

    ssr_total = 0

    # Time point 24
    T_pred = y1[1, 0] + y1[1, 1] + y1[1, 2]
    mask = ~np.isnan(T_data_list[0])
    ssr_total += np.sum((T_pred - T_data_list[0][mask]) ** 2)

    # Time point 25.04167 - Virus data
    valid_v = ~np.isnan(V_data_list[0])
    if np.any(valid_v):
        logV_pred = safe_log10(y2[1, 3])
        logV_data = safe_log10(V_data_list[0][valid_v])
        valid = ~np.isnan(logV_pred) & ~np.isnan(logV_data)
        if np.any(valid):
            ssr_total += np.sum((logV_pred - logV_data[valid]) ** 2)

    # Time point 26
    T_pred = y2[2, 0] + y2[2, 1] + y2[2, 2]
    if np.any(valid_v):
        logV_data = safe_log10(V_data_list[0][valid_v])
        valid = ~np.isnan(logV_data)
        if np.any(valid):
            ssr_total += np.sum((T_pred - logV_data[valid]) ** 2)

    # Time point 27
    mask = ~np.isnan(T_data_list[1])
    ssr_total += np.sum((y2[3, 0] + y2[3, 1] + y2[3, 2] - T_data_list[1][mask]) ** 2)

    # Time point 28 - Virus data
    valid_v = ~np.isnan(V_data_list[1])
    if np.any(valid_v):
        logV_pred = safe_log10(y3[1, 3])
        logV_data = safe_log10(V_data_list[1][valid_v])
        valid = ~np.isnan(logV_pred) & ~np.isnan(logV_data)
        if np.any(valid):
            ssr_total += np.sum((logV_pred - logV_data[valid]) ** 2)

    # Time point 29
    mask = ~np.isnan(T_data_list[2])
    ssr_total += np.sum((y3[2, 0] + y3[2, 1] + y3[2, 2] - T_data_list[2][mask]) ** 2)

    # Time point 31
    mask = ~np.isnan(T_data_list[3])
    ssr_total += np.sum((y4[0, 0] + y4[0, 1] + y4[0, 2] - T_data_list[3][mask]) ** 2)

    # Time point 33 - Virus data
    valid_v = ~np.isnan(V_data_list[2])
    if np.any(valid_v):
        logV_pred = safe_log10(y4[1, 3])
        logV_data = safe_log10(V_data_list[2][valid_v])
        valid = ~np.isnan(logV_pred) & ~np.isnan(logV_data)
        if np.any(valid):
            ssr_total += np.sum((logV_pred - logV_data[valid]) ** 2)

    # Remaining time points (35, 38, 40, 42, 45, 47, 49, 52, 54, 56, 59, 61, 63, 66)
    time_indices = [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for i, idx in enumerate(time_indices):
        data_idx = i + 4  # T_data_list index
        if data_idx < len(T_data_list):
            mask = ~np.isnan(T_data_list[data_idx])
            T_pred = y4[idx, 0] + y4[idx, 1] + y4[idx, 2]
            ssr_total += np.sum((T_pred - T_data_list[data_idx][mask]) ** 2)

    # Return a large value if ssr_total is NaN to help the optimizer
    return float(ssr_total if not np.isnan(ssr_total) else 1e10)


# Parameter bounds and initial guess
initial_guess = [0.09, np.log10(1e-10), 0.2, 0.1, np.log10(1e5), 0.1]
bounds = [
    (0.001, 0.5),
    (np.log10(1e-11), np.log10(1e-5)),
    (0.05, 5.0),
    (0.05, 5.0),
    (np.log10(1e2), np.log10(1e8)),
    (0.05, 10.0)
]

# Optimization
result = minimize(ssr, initial_guess, bounds=bounds, method='L-BFGS-B')

# Print fitted parameters
λ_fit, β_log, k_fit, δ_fit, p_log, c_fit = result.x
β_fit = 10 ** β_log
p_fit = 10 ** p_log
print(f"Fitted λ: {λ_fit:.4f}")
print(f"Fitted β: {β_fit:.4e}")
print(f"Fitted k: {k_fit:.4f}")
print(f"Fitted δ: {δ_fit:.4f}")
print(f"Fitted p: {p_fit:.4e}")
print(f"Fitted c: {c_fit:.4f}")


# Generate predictions for plotting
def generate_predictions(params, t_eval):
    λ, β_log, k, δ, p_log, c = params
    β = 10 ** β_log
    p = 10 ** p_log
    y0 = [1.76, 0, 0, 0]

    # Find where to add virus
    add_points = [25, 28, 31]
    segments = []
    current_t = 0
    current_y = y0

    for t_add in sorted(add_points + [t_eval.max()]):
        if t_add > current_t:
            t_segment = t_eval[(t_eval >= current_t) & (t_eval <= t_add)]
            if len(t_segment) > 0:
                y_segment = odeint(base_model, current_y, t_segment, args=(λ, β, k, δ, p, c))
                segments.append((t_segment, y_segment))
                current_y = y_segment[-1, :].copy()
                if t_add in add_points:
                    current_y[3] += 100  # Add virus
        current_t = t_add

    # Combine all segments
    all_t = np.concatenate([s[0] for s in segments])
    all_y = np.concatenate([s[1] for s in segments])
    T_pred = all_y[:, 0] + all_y[:, 1] + all_y[:, 2]

    return all_t, T_pred


# Create fine time grid for smooth plot
t_pred = np.linspace(0, 66, 300)
t_pred_fine, T_pred = generate_predictions(result.x, t_pred)

# Plotting
plt.figure(figsize=(10, 6))

# Plot data points (handling NaNs)
for i in range(len(t_data)):
    if i < len(T_data_list):
        mask = ~np.isnan(T_data_list[i])
        plt.scatter(
            np.full(np.sum(mask), t_data[i]),
            T_data_list[i][mask],
            s=50, color='orangered', alpha=0.7,
            label="Data" if i == 0 else ""
        )

# Plot model fit
plt.plot(t_pred_fine, T_pred, 'b-', linewidth=2, label='Model Fit')

plt.title("Tumor Growth Model Fit", fontsize=14)
plt.ylabel('Tumor Volume (mm³)', fontsize=12)
plt.xlabel('Time (days)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()