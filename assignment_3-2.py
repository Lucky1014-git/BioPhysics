import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Step 1: Input data
days = np.array([0, 1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36, 38, 41])
virus_data = [
    [111, 115, 115, 148, 102],
    [120, 123, 123, 156, 110],
    [131, 132, 127, 168, 116],
    [141, 144, 136, 183, 126],
    [154, 154, 148, 202, 138],
    [172, 179, 166, 216, 175],
    [185, 195, 181, 237, 191],
    [196, 207, 195, 254, 207],
    [215, 219, 204, 269, 222],
    [359, 413, 310, 370, 285],
    [521, 447, 352, 447, 345],
    [546, 470, 371, 472, 364],
    [708, 599, 510, 661, 638],
    [759, 637, 538, 690, 674],
    [853, 993, 637, 783, 759],
    [1021, np.nan, 759, 985, np.nan],
    [933, 1231, np.nan, np.nan, np.nan],
    [1001, np.nan, np.nan, np.nan, np.nan],
    [1247, np.nan, np.nan, np.nan, np.nan]
]

virus_data = np.array(virus_data, dtype=np.float64)
virus_avg = np.nanmean(virus_data, axis=1)

# Step 2: Virus model
def virus_model(y, t, λ, β, k, δ, p, c):
    T, E, I, V = y
    dTdt = λ * T - β * T * V
    dEdt = β * T * V - k * E
    dIdt = k * E - δ * I
    dVdt = p * I - c * V
    return [dTdt, dEdt, dIdt, dVdt]

# SSR function for optimization
def compute_ssr(params, t, y_data, y0):
    λ, β, k, δ, p, c = params
    sol = odeint(virus_model, y0, t, args=(λ, β, k, δ, p, c))
    V = sol[:, 3]
    return np.sum((V - y_data) ** 2)

# Fit function per segment
def fit_segment(t_local, y_data, y0, bounds):
    initial_guess = [0.5, 0.001, 0.1, 0.1, 1.0, 0.1]
    result = minimize(compute_ssr, initial_guess, args=(t_local, y_data, y0), bounds=bounds)
    optimal_params = result.x
    ssr_value = result.fun
    sol = odeint(virus_model, y0, t_local, args=tuple(optimal_params))
    return optimal_params, sol, ssr_value

# Segment definitions
segments = [(0, 3), (3, 6), (6, 41)]
param_bounds = [(0, 2), (0, 0.01), (0, 1), (0, 1), (0, 10), (0, 1)]

# Initial conditions
T0, E0, I0, V0 = 1.76, 0.0, 0.0, 100.0
segment_solutions = []
segment_params = []
segment_ssrs = []

# Loop over segments to fit model
for i, (start, end) in enumerate(segments):
    mask = (days >= start) & (days <= end)
    x_seg = days[mask]
    y_seg = virus_avg[mask]

    t_local = x_seg - x_seg[0]  # reset segment time to start at zero
    params, sol, ssr = fit_segment(t_local, y_seg, [T0, E0, I0, V0], param_bounds)

    print(f"\n--- Segment {i+1}: Days {start}-{end} ---")
    print(f"Optimal Parameters: λ={params[0]:.5f}, β={params[1]:.6f}, k={params[2]:.5f}, δ={params[3]:.5f}, p={params[4]:.5f}, c={params[5]:.5f}")
    print(f"SSR: {ssr:.4f}")

    # Prepare initial conditions for next segment
    T0, E0, I0, V0 = sol[-1]
    V0 += 100  # increment virus as you wanted
    segment_params.append(params)
    segment_solutions.append((x_seg, sol))
    segment_ssrs.append(ssr)

# --------- UPDATED PLOTTING ---------
plt.figure(figsize=(10, 6))
plt.plot(days, virus_avg, 'ko', label='Observed Data')

for i, ((start, end), params, (x_seg, sol)) in enumerate(zip(segments, segment_params, segment_solutions)):
    # Dense time grid for smooth plotting in the segment
    t_dense = np.linspace(0, x_seg[-1] - x_seg[0], 100)
    y0_dense = sol[0]
    sol_dense = odeint(virus_model, y0_dense, t_dense, args=tuple(params))
    t_dense_abs = t_dense + x_seg[0]

    plt.plot(t_dense_abs, sol_dense[:, 3], label=f'Model Fit: Days {start}-{end}')

plt.xlabel("Days Post Infection")
plt.ylabel("Virus Concentration (V)")
plt.title("Viral Dynamics Model Fit (Smooth Curves)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
