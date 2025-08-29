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
    np.array([733, np.nan,np.nan, np.nan, 570]),
    np.array([846, np.nan, np.nan, np.nan, 617]),
    np.array([922, np.nan, np.nan, np.nan,661]),
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
    ssr += (y1[1, 0] + y1[1, 1] + y1[1, 2] - T_data_list[0]) ** 2  # t = 24

    # Handle potential log10(0) or log10(negative) cases
    v1 = y2[1, 3]
    log_v1 = np.log10(v1) if v1 > 0 else -np.inf  # or use a very small number like -20
    ssr += (log_v1 - np.log10(V_data_list[0])) ** 2 if np.isfinite(log_v1) else 0

    ssr += (y2[1, 0] + y2[1, 1] + y2[1, 2] - np.log10(V_data_list[0])) ** 2  # t = 26
    ssr += (y2[1, 0] + y2[1, 1] + y2[1, 2] - T_data_list[1]) ** 2  # t = 27
    ssr += (y3[1, 0] + y3[1, 1] + y3[1, 2] - np.log10(V_data_list[1])) ** 2  # t = 28
    ssr += (y3[1, 0] + y3[1, 1] + y3[1, 2] - T_data_list[2]) ** 2  # t = 29
    ssr += (y4[1, 0] + y4[1, 1] + y4[1, 2] - T_data_list[3]) ** 2  # t = 31
    ssr += (y4[1, 0] + y4[1, 1] + y4[1, 2] - np.log10(V_data_list[2])) ** 2  # t = 33
    ssr += (y4[1, 0] + y4[1, 1] + y4[1, 2] - T_data_list[4]) ** 2  # t = 35
    ssr += (y4[1, 0] + y4[1, 1] + y4[1, 2] - T_data_list[5]) ** 2  # t = 38
    ssr += (y4[1, 0] + y4[1, 1] + y4[1, 2] - T_data_list[6]) ** 2  # t = 40
    ssr += (y4[1, 0] + y4[1, 1] + y4[1, 2] - T_data_list[7]) ** 2  # t = 42
    ssr += (y4[1, 0] + y4[1, 1] + y4[1, 2] - T_data_list[8]) ** 2  # t = 45
    ssr += (y4[1, 0] + y4[1, 1] + y4[1, 2] - T_data_list[9]) ** 2  # t = 47
    ssr += (y4[1, 0] + y4[1, 1] + y4[1, 2] - T_data_list[10]) ** 2  # t = 49
    ssr += (y4[1, 0] + y4[1, 1] + y4[1, 2] - T_data_list[11]) ** 2  # t = 52
    ssr += (y4[1, 0] + y4[1, 1] + y4[1, 2] - T_data_list[12]) ** 2  # t = 54
    ssr += (y4[1, 0] + y4[1, 1] + y4[1, 2] - T_data_list[13]) ** 2  # t = 56
    ssr += (y4[1, 0] + y4[1, 1] + y4[1, 2] - T_data_list[14]) ** 2  # t = 59
    ssr += (y4[1, 0] + y4[1, 1] + y4[1, 2] - T_data_list[15]) ** 2  # t = 61
    ssr += (y4[1, 0] + y4[1, 1] + y4[1, 2] - T_data_list[16]) ** 2  # t = 63
    ssr += (y4[1, 0] + y4[1, 1] + y4[1, 2] - T_data_list[17]) ** 2  # t = 66

    return float(np.sum(ssr))

# For parameters spanning orders of magnitude (β, p)
initial_guess = [0.09, np.log10(1e-10), 0.2, 0.1, np.log10(1e5), 0.1]
bounds = [
    (0.001, 0.5),
    (np.log10(1e-11), np.log10(1e-5)),  # β in log space
    (0.05, 5.0),
    (0.05, 5.0),
    (np.log10(1e2), np.log10(1e8)),     # p in log space
    (0.05, 10.0)
]

result = minimize(ssr, initial_guess,
                  bounds=bounds, method='nelder-mead')

λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit = result.x
print(f"Fitted λ: {λ_fit:.4f}")
print(f"Fitted β: {β_fit:.4f}")
print(f"Fitted k: {k_fit:.4f}")
print(f"Fitted δ: {δ_fit:.4f}")
print(f"Fitted p: {p_fit:.4f}")
print(f"Fitted c: {c_fit:.4f}")

T_pred_full = []
t_pred_full = []

y0 = [1.76, 0, 0, 0]
y1 = odeint(base_model, y0, np.linspace(0, 25, 100), args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))
y02 = y1[-1, :]
y02[3] += 100  # Add virus
# --- Segment 2: 25–28 ---
y2 = odeint(base_model, y02, np.linspace(25, 28, 100), args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))
y03 = y2[-1, :]
y03[3] += 100  # Add virus again
# --- Segment 3: 28–31 ---
y3 = odeint(base_model, y03, np.linspace(28, 31, 100), args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))
y04 = y3[-1, :]
y04[3] += 100  # Add virus again
# --- Segment 4: 31–41 ---
y4 = odeint(base_model, y04, np.linspace(31, 66, 100), args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))
T_pred_full.extend(y1[:, 0])
t_pred_full.extend(np.linspace(0, 66, 100))

T_pred = np.array(T_pred_full)
t_pred = np.array(t_pred_full)

#for T_data in T_data_list:
   # mask = ~np.isnan(T_data)
   # plt.scatter(t_data[mask], T_data[mask], s=8, color='orangered', label="Data" if T_data is T_data_list[0] else "")
mask_list = [~np.isnan(arr) for arr in T_data_list]

for i in range(len(t_data)):
    if i < len(T_data_list):
        plt.scatter(
            np.full(np.sum(mask_list[i]), t_data[i]),     # repeated x-values
            T_data_list[i][mask_list[i]],                 # filtered y-values
            s=8, color='orangered'
        )


plt.plot(t_pred, T_pred, color='blue', label='Best Fit')
plt.title("Tumor Growth (Model Fit)")
plt.ylabel('Tumor volume(mm³)')
plt.xlabel('Time (days)')
plt.legend()
plt.grid()
plt.show()