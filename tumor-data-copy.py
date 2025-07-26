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


def ssr(params):
    λ, β, k, δ, p, c = params
    y0 = [1.76, 0, 0, 0]
    y1 = odeint(base_model, y0, [0, 24, 25], args=(λ, β, k, δ, p, c))
    y02 = y1[-1, :]
    y02[3] = y02[3] + 100
    y2 = odeint(base_model, y02, [25, 25.04167, 26, 27, 28], args=(λ, β, k, δ, p, c))
    y03 = y2[-1, :]
    y03[3] = y03[3] + 100
    y3 = odeint(base_model, y03, [28, 29, 31], args=(λ, β, k, δ, p, c))
    y04 = y3[-1, :]
    y04[3] = y04[3] + 100
    y4 = odeint(base_model, y04, [31, 33, 35, 38, 40, 42, 45, 47, 49, 52, 54, 56, 59, 61, 63, 66],
                args=(λ, β, k, δ, p, c))

    ssr = 0
    ssr += np.sum((y1[1, 0] + y1[1, 1] + y1[1, 2] - T_data_list[0]) ** 2)  # t = 24
    ssr += np.sum((np.log10(y2[1, 3]) - np.log10(V_data_list[0])) ** 2)  # t = 25.04167
    ssr += np.sum((y2[2, 0] + y2[2, 1] + y2[2, 2] - T_data_list[1]) ** 2)  # t = 26
    ssr += np.sum((y2[3, 0] + y2[3, 1] + y2[3, 2] - T_data_list[1]) ** 2)  # t = 27
    ssr += np.sum((np.log10(y3[1, 3]) - np.log10(V_data_list[1])) ** 2)  # t = 28
    ssr += np.sum((y3[2, 0] + y3[2, 1] + y3[2, 2] - T_data_list[2]) ** 2)  # t = 29
    ssr += np.sum((y4[0, 0] + y4[0, 1] + y4[0, 2] - T_data_list[3]) ** 2)  # t = 31
    ssr += np.sum((np.log10(y4[2, 3]) - np.log10(V_data_list[2])) ** 2)  # t = 33
    # ... (rest of your SSR terms)
    return float(np.sum(ssr))


# For parameters spanning orders of magnitude (β, p)
initial_guess = [0.1, np.log10(1e-8), 0.1, 0.1, np.log10(1e5), 0.1]
bounds = [
    (0.001, 0.5),
    (np.log10(1e-11), np.log10(1e-5)),  # β in log space
    (0.05, 5.0),
    (0.05, 5.0),
    (np.log10(1e2), np.log10(1e8)),  # p in log space
    (0.05, 10.0)
]

result = minimize(ssr, initial_guess, bounds=bounds, method='nelder-mead')

# Convert back from log space where needed
λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit = result.x
β_fit = 10 ** β_fit
p_fit = 10 ** p_fit

print(f"Fitted λ: {λ_fit:.4f}")
print(f"Fitted β: {β_fit:.4e}")
print(f"Fitted k: {k_fit:.4f}")
print(f"Fitted δ: {δ_fit:.4f}")
print(f"Fitted p: {p_fit:.4e}")
print(f"Fitted c: {c_fit:.4f}")

# Create full prediction for plotting
t_full = np.linspace(0, 66, 500)
y0 = [1.76, 0, 0, 0]

# First segment: 0-25 days (before first virus addition)
y1 = odeint(base_model, y0, t_full[t_full <= 25], args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))

# Second segment: 25-28 days (after first virus addition)
y02 = y1[-1, :].copy()
y02[3] += 100
y2 = odeint(base_model, y02, t_full[(t_full > 25) & (t_full <= 28)], args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))

# Third segment: 28-31 days (after second virus addition)
y03 = y2[-1, :].copy()
y03[3] += 100
y3 = odeint(base_model, y03, t_full[(t_full > 28) & (t_full <= 31)], args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))

# Fourth segment: 31-66 days (after third virus addition)
y04 = y3[-1, :].copy()
y04[3] += 100
y4 = odeint(base_model, y04, t_full[t_full > 31], args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))

# Combine all segments
T_pred = np.concatenate([y1[:, 0], y2[:, 0], y3[:, 0], y4[:, 0]])
V_pred = np.concatenate([y1[:, 3], y2[:, 3], y3[:, 3], y4[:, 3]])

# Create figure with two subplots
plt.figure(figsize=(12, 6))

# Tumor plot
plt.subplot(1, 2, 1)
mask_list = [~np.isnan(arr) for arr in T_data_list]
for i in range(len(t_data)):
    if i < len(T_data_list):
        plt.scatter(
            np.full(np.sum(mask_list[i]), t_data[i]),
            T_data_list[i][mask_list[i]],
            s=8, color='orangered'
        )
plt.plot(t_full, T_pred, color='blue', label='Best Fit')
plt.title("Tumor Growth (Model Fit)")
plt.ylabel('Tumor volume (mm³)')
plt.xlabel('Time (days)')
plt.legend()
plt.grid()

# Virus plot (log scale)
plt.subplot(1, 2, 2)
# Virus measurement times (assuming these correspond to 25.04167, 28, and 33 days)
v_times = [25.04167, 28, 33]
for i, (vt, vd) in enumerate(zip(v_times, V_data_list)):
    mask = ~np.isnan(vd)
    plt.scatter(
        np.full(np.sum(mask), vt),
        vd[mask],
        s=8, color='green', label='Virus Data' if i == 0 else ""
    )
plt.plot(t_full, V_pred, color='purple', label='Virus Fit')
plt.yscale('log')
plt.title("Viral Load (Model Fit)")
plt.ylabel('Virus concentration (log scale)')
plt.xlabel('Time (days)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()