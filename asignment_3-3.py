import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Time points and virus data (5 replicates per time)
time_points = np.array([0, 1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36, 38, 41])
virus_data = np.array([
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
    [np.nan, np.nan, 933, 1231, np.nan],
    [np.nan, np.nan, 1001, np.nan, np.nan],
    [np.nan, np.nan, 1247, np.nan, np.nan]
])
virus_mean = np.nanmean(virus_data, axis=1)

# ODE system
def model(y, t, λ, β, k, δ, p, c):
    T, E, I, V = y
    dTdt = λ * T - β * T * V
    dEdt = β * T * V - k * E
    dIdt = k * E - δ * I
    dVdt = p * I - c * V
    return [dTdt, dEdt, dIdt, dVdt]

# SSR function for optimization
def ssr(params, t, y_data, y0):
    λ, β, k, δ, p, c = params
    y_model = odeint(model, y0, t, args=(λ, β, k, δ, p, c))
    V_model = y_model[:, 0]

    # Take log of both observed and predicted virus values
    log_y_data = np.log(y_data)  # small value to avoid log(0)
    log_V_model = np.log(V_model)  # small value to avoid log(0)

    return np.sum((log_y_data - log_V_model) ** 2)


# Fit helper function
def fit_segment(t_seg, v_seg, y0):
    result = minimize(ssr, initial_guess, args=(t_seg, v_seg, y0), bounds=bounds)
    best_params = result.x
    y_fit = odeint(model, y0, t_seg, args=tuple(best_params))
    return best_params, y_fit

# Initial guess and bounds for optimization
initial_guess = [0.1, 1e-5, 0.1, 0.1, 1e5, 0.1]  # λ, β, k, δ, p, c
bounds = [
    (0.0001, 1),     # λ
    (0, 1e-2),       # β
    (0.0001, 1),     # k
    (0.0001, 1),     # δ
    (1e3, 1e7),      # p
    (0.0001, 1)      # c
]

# Time segments for fitting
segments = [(0, 3), (3, 6), (6, 41)]
# Segment virus data properly (averaging virus data in each segment)
t_segments = [time_points[(time_points >= start) & (time_points <= end)] for start, end in segments]
v_segments = [np.mean(virus_mean[(time_points >= start) & (time_points <= end)]) for start, end in segments]

# Initial conditions
T0, E0, I0, V0 = 1.76, 0, 0, 100
y0 = [T0, E0, I0, V0]

# Store results
all_t, all_T = [], []
current_y0 = y0

# Iterate through each segment
for t_seg, v_avg in zip(t_segments, v_segments):
    # Fit using the average virus value for this segment
    best_params, y_fit = fit_segment(t_seg, v_avg, current_y0)

    print(f"Best Params: {best_params}, SSR: {ssr(best_params, t_seg, v_avg, current_y0)}")

    # Store time and T results
    all_t.extend(t_seg)
    all_T.extend(y_fit[:, 0])

    # Update initial condition for next segment
    current_y0 = y_fit[-1]
    current_y0[3] += 100  # Add 100 to V for next segment (example logic)


# Convert to arrays
all_t = np.array(all_t)
all_T = np.array(all_T)
replicates = virus_data[:, 0:].T

plt.figure(figsize=(10, 5))

# Create the plot
for i, rep in enumerate(replicates):
    mask = ~np.isnan(rep)  # filter out NaNs
    plt.scatter(time_points[mask], rep[mask], s=20, color='red', label=f'Replicate', alpha=0.8)

# Plot results
plt.plot(all_t, all_T, label='Fitted Virus Load (V)', marker='o')
#plt.scatter(time_points, virus_mean, label='Observed Virus Data', color='red')
plt.xlabel("Days Post Infection")
plt.ylabel("Virus Load")
plt.title("Fitted Virus Model vs Observed Data")
plt.legend()
plt.grid(True)
plt.show()
