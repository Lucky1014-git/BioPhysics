import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- Helper Function ---
def safe_log10(x):
    return np.log10(np.maximum(x, 1e-10))

# --- Experimental Data ---
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

# --- Model ---
def base_model(Y, t, λ, β, k, δ, p, c):
    T, E, I, V = Y
    dTdt = λ * T - β * T * V
    dEdt = β * T * V - k * E
    dIdt = k * E - δ * I
    dVdt = p * I - c * V
    return [dTdt, dEdt, dIdt, dVdt]

# --- SSR Function ---
def ssr(params):
    λ, log_β, k, δ, log_p, c = params
    β = 10 ** log_β
    p = 10 ** log_p

    total_ssr = 0.0
    y0 = [1.76, 0, 0, 0]

    # Segment 1: 0–26 (no virus injection yet)
    t1 = [0, 24, 26]
    y1 = odeint(base_model, y0, t1, args=(λ, β, k, δ, p, c))
    total_ssr += np.nansum((T_data_list[0] - np.sum(y1[1, :3])) ** 2)  # t=24

    # Inject virus at t=26
    y2_init = y1[-1]
    y2_init[3] += 100  # Add 100 units of virus

    # Segment 2: 26–28
    t2 = [26, 26.04167, 27, 28]
    y2 = odeint(base_model, y2_init, t2, args=(λ, β, k, δ, p, c))

    total_ssr += np.nansum((safe_log10(y2[1, 3]) - safe_log10(V_data_list[0])) ** 2)  # t=26.04167
    total_ssr += np.nansum((T_data_list[1] - np.sum(y2[2, :3])) ** 2)  # t=26
    total_ssr += np.nansum((T_data_list[2] - np.sum(y2[3, :3])) ** 2)  # t=28

    # Inject virus again at t=28
    y3_init = y2[-1]
    y3_init[3] += 100

    # Segment 3: 28–31
    t3 = [28, 29, 31]
    y3 = odeint(base_model, y3_init, t3, args=(λ, β, k, δ, p, c))
    total_ssr += np.nansum((safe_log10(y3[0, 3]) - safe_log10(V_data_list[1])) ** 2)  # t=28
    total_ssr += np.nansum((T_data_list[3] - np.sum(y3[1, :3])) ** 2)  # t=29
    total_ssr += np.nansum((T_data_list[4] - np.sum(y3[2, :3])) ** 2)  # t=31

    # Inject virus again at t=31
    y4_init = y3[-1]
    y4_init[3] += 100

    # Segment 4: 31–66
    t4 = [31, 33, 35, 38, 40, 42, 45, 47, 49, 52, 54, 56, 59, 61, 63, 66]
    y4 = odeint(base_model, y4_init, t4, args=(λ, β, k, δ, p, c))
    for i, ti in enumerate(t4):
        if i + 5 < len(T_data_list):
            total_ssr += np.nansum((T_data_list[i + 5] - np.sum(y4[i, :3])) ** 2)

    total_ssr += np.nansum((safe_log10(y4[1, 3]) - safe_log10(V_data_list[2])) ** 2)  # t=33

    print(f"SSR: {total_ssr:.4f}")
    return total_ssr

# --- Optimization ---
initial_guess = [0.09, np.log10(1e-10), 0.2, 0.1, np.log10(1e5), 0.1]
bounds = [
    (0.001, 0.5),
    (np.log10(1e-11), np.log10(1e-5)),
    (0.05, 5.0),
    (0.05, 5.0),
    (np.log10(1e2), np.log10(1e8)),
    (0.05, 10.0)
]

result = minimize(ssr, initial_guess, bounds=bounds, method='nelder-mead')

λ_fit, β_fit_log, k_fit, δ_fit, p_fit_log, c_fit = result.x
β_fit = 10 ** β_fit_log
p_fit = 10 ** p_fit_log

print(f"Fitted λ: {λ_fit:.4f}")
print(f"Fitted β: {β_fit:.4e}")
print(f"Fitted k: {k_fit:.4f}")
print(f"Fitted δ: {δ_fit:.4f}")
print(f"Fitted p: {p_fit:.4e}")
print(f"Fitted c: {c_fit:.4f}")

# --- Plotting ---
T_pred_full = []
t_pred_full = []

y0 = [1.76, 0, 0, 0]
t_all = []

# Segment 1: 0–26
t1 = np.linspace(0, 26, 50)
y1 = odeint(base_model, y0, t1, args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))
y = y1
y[-1, 3] += 100  # Add virus at t=26

# Segment 2: 26–28
t2 = np.linspace(26, 28, 50)
y2 = odeint(base_model, y[-1], t2, args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))
y = np.vstack((y, y2))
y[-1, 3] += 100  # Add virus at t=28

# Segment 3: 28–31
t3 = np.linspace(28, 31, 50)
y3 = odeint(base_model, y[-1], t3, args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))
y = np.vstack((y, y3))
y[-1, 3] += 100  # Add virus at t=31

# Segment 4: 31–66
t4 = np.linspace(31, 66, 100)
y4 = odeint(base_model, y[-1], t4, args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))
y = np.vstack((y, y4))

t_pred = np.concatenate((t1, t2, t3, t4))
T_pred = y[:, 0]

# Plot tumor data
mask_list = [~np.isnan(arr) for arr in T_data_list]
for i in range(len(t_data)):
    if i < len(T_data_list):
        plt.scatter(
            np.full(np.sum(mask_list[i]), t_data[i]),
            T_data_list[i][mask_list[i]],
            s=8, color='orangered'
        )

plt.plot(t_pred, y[:, 0], color='blue', label='Best Fit')
plt.title("Tumor Growth (Model Fit)")
plt.ylabel('Tumor volume (mm³)')
plt.xlabel('Time (days)')
plt.legend()
plt.grid()
plt.show()