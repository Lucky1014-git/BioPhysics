import numpy as np
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

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
intialVirusCount = 100
y0 = [T0, 0, 0, intialVirusCount]  # [T, E, I, V]

# Model
def base_model(Y, t, λ, β, k, δ, p, c):
    T, E, I, V = Y
    dTdt = λ * T - β * T * V
    dEdt = β * T * V - k * E
    dIdt = k * E - δ * I
    dVdt = p * I - c * V
    return [dTdt, dEdt, dIdt, dVdt]


def ssr_segment_0_24(params):
    λ, β, k, δ, p, c = params
    total_ssr = 0

    # Define time masks for each segment
    t_mask_0_24 = (t_data >= 0) & (t_data <= 24)
 #   t_mask_3_6 = (t_data >= 3) & (t_data <= 6)
 #   t_mask_6_41 = (t_data >= 6) & (t_data <= 41)

    t_seg_0_24 = t_data[t_mask_0_24]
  #   t_seg_3_6 = t_data[t_mask_3_6]
  #  t_seg_6_41 = t_data[t_mask_6_41]

    for T_data in T_data_list:
        try:
            # --- Segment 1: (0–24) ---
            T_0_24 = T_data[t_mask_0_24]
            mask_0_24 = ~np.isnan(T_0_24)
            sol_0_24 = odeint(base_model, y0, t_seg_0_24[mask_0_24], args=(λ, β, k, δ, p, c))
            T_model_0_24 = sol_0_24[:, 0]
            total_ssr += np.sum((T_0_24[mask_0_24] - T_model_0_24) ** 2)

        except Exception as e:
            print(f"Error during fitting for a replicate: {e}")
            return np.inf

    return total_ssr
#def ssr_segment_0_41(params):
   # return ssr_segment_0_3(params) + ssr_segment_3_6(params) + ssr_segment_6_41(params)


# Initial guess and bounds:  λ, β, k, δ, p, c
initial_guess = [0.1, 1e-7, 0.1, 0.1, 1e4, 0.1]
bounds = [
    (0.01, 0.5),     # λ - narrower range
    (1e-8, 1e-7),    # β - narrower range
    (0.01, 0.5),     # k - narrower range
    (0.1, 1.0),      # δ - narrower range
    (1e3, 1e5),      # p - narrower range
    (0.1, 1.0)       # c - narrower range
]


# Fit parameters with faster settings
result = differential_evolution(ssr_segment_0_24, bounds,
                              seed=42,
                              maxiter=100,      # Reduced from 1000
                              popsize=10,       # Small population
                              tol=1e-4,         # Less strict tolerance
                              atol=1e-4,        # Less strict absolute tolerance
                              workers=1,        # Single worker for stability
                              disp=True)        # Show progress


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
t_seg_0_24 = t_data[(t_data >= 0) & (t_data <= 24)]
y0_0_24 = [T0, 0, 0, 25]
sol_0_24 = odeint(base_model, y0_0_24, t_seg_0_24, args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))
T_pred_full.extend(sol_0_24[:, 0])
t_pred_full.extend(t_seg_0_24)
#y0_3_6 = sol_0_24[-1].copy()
#y0_3_6[3] += 100  # Add virus

# --- Segment 2: 3–6 ---
#t_seg_3_6 = t_data[(t_data > 3) & (t_data <= 6)]  # exclude 3 to avoid duplication
#sol_3_6 = odeint(base_model, y0_3_6, t_seg_3_6, args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))
#T_pred_full.extend(sol_3_6[:, 0])
#t_pred_full.extend(t_seg_3_6)
#y0_6_41 = sol_3_6[-1].copy()
#y0_6_41[3] += 100  # Add virus again

# --- Segment 3: 6–41 ---
#t_seg_6_41 = t_data[(t_data > 6)]  # exclude 6 to avoid duplication
#sol_6_41 = odeint(base_model, y0_6_41, t_seg_6_41, args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))
#T_pred_full.extend(sol_6_41[:, 0])
#t_pred_full.extend(t_seg_6_41)

T_pred = np.array(T_pred_full)
t_pred = np.array(t_pred_full)

# Predict using best-fit parameters
#PBSpred = odeint(base_model, y0, t_data, args=tuple(result.x))
#T_pred = PBSpred[:, 0]

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.35)

# Create slider
ax_slider = plt.axes([0.2, 0.15, 0.5, 0.03])
slider = Slider(ax_slider, 'Initial Virus Count', 1, 100, valinit=intialVirusCount, valfmt='%d')

# Create reload button
ax_button = plt.axes([0.4, 0.05, 0.1, 0.04])
button = Button(ax_button, 'Reload')

# Initial plot
for T_data in T_data_list:
    mask = ~np.isnan(T_data)
    ax.scatter(t_data[mask], T_data[mask], s=8, color='orangered', label="Data" if T_data is T_data_list[0] else "")

line, = ax.plot(t_pred, T_pred, color='blue', label='Best Fit')

def update_plot(val):
    # Get current slider value
    current_virus_count = int(slider.val)
    print(f"Current Virus Count: {current_virus_count}")

    # Update initial conditions
    y0_updated = [T0, 0, 0, current_virus_count]

    # Recalculate prediction with new initial virus count
    t_seg_0_24 = t_data[(t_data >= 0) & (t_data <= 24)]
    sol_0_24 = odeint(base_model, y0_updated, t_seg_0_24, args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))
    T_pred_updated = sol_0_24[:, 0]

    print(f"Updated T range: {T_pred_updated.min():.4f} to {T_pred_updated.max():.4f}")

    # Update both x and y data for the plot
    line.set_data(t_seg_0_24, T_pred_updated)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    fig.canvas.flush_events()

def reload_plot(event):
    update_plot(slider.val)

# Connect slider and button to update functions
slider.on_changed(update_plot)
button.on_clicked(reload_plot)

ax.set_title("Tumor Growth (Model Fit)")
ax.set_ylabel('Tumor volume(mm³)')
ax.set_xlabel('Time (days)')
ax.legend()
ax.grid()
plt.show()