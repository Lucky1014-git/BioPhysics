import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Experimental data
t_data = np.array([0, 1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36, 38, 41])
T_data_list = [
    np.array([111, 120, 131, 141, 154, 172, 185, 196, 215, 359, 521, 546, 708, 759, 853, 1021, np.nan, np.nan, np.nan]),
    np.array([115, 123, 132, 144, 154, 179, 195, 207, 219, 413, 447, 470, 599, 637, 993, np.nan, np.nan, np.nan, np.nan]),
    np.array([115, 123, 127, 136, 148, 166, 181, 195, 204, 310, 352, 371, 510, 538, 637, 759, 933, 1001, 1247]),
    np.array([148, 156, 168, 183, 202, 216, 237, 254, 269, 370, 447, 472, 661, 690, 783, 985, 1231, np.nan, np.nan]),
    np.array([102, 110, 116, 126, 138, 175, 191, 207, 222, 285, 345, 364, 638, 674, 759, np.nan, np.nan, np.nan, np.nan])
]

# Initial conditions and parameters
T0 = 100
V0 = 100
y0 = [T0, 0, 0, V0]

# Optimized starting parameters for better initial fit
params = {
    'λ': 0.12,    # Growth rate
    'β': 3.5e-7,  # Infection rate
    'k': 0.25,    # Transition rate
    'δ': 0.4,     # Death rate
    'p': 4.2e5,   # Virus production
    'c': 0.35     # Clearance rate
}

# Model function
def base_model(Y, t, λ, β, k, δ, p, c):
    T, E, I, V = Y
    dTdt = λ * T - β * T * V
    dEdt = β * T * V - k * E
    dIdt = k * E - δ * I
    dVdt = p * I - c * V
    return [dTdt, dEdt, dIdt, dVdt]

# Create figure and axes
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(bottom=0.4)

# Plot experimental data
for T_data in T_data_list:
    mask = ~np.isnan(T_data)
    ax.scatter(t_data[mask], T_data[mask], s=30, color='orangered', alpha=0.7)

# Initial model prediction
t_fine = np.linspace(0, 41, 200)
sol = odeint(base_model, y0, t_fine, args=tuple(params.values()))
line, = ax.plot(t_fine, sol[:, 0], 'b-', linewidth=2, label='Model')

# Slider setup
slider_axes = []
sliders = []
param_ranges = {
    'λ': (0.01, 0.3),
    'β': (1e-8, 1e-5),
    'k': (0.05, 0.8),
    'δ': (0.1, 1.0),
    'p': (1e4, 1e7),
    'c': (0.05, 1.0)
}

# Create sliders with optimized starting positions
for i, (name, (min_val, max_val)) in enumerate(param_ranges.items()):
    ax_slider = plt.axes([0.25, 0.3 - 0.05*i, 0.65, 0.03])
    slider = Slider(
        ax=ax_slider,
        label=name,
        valmin=min_val,
        valmax=max_val,
        valinit=params[name],  # Set to optimized starting value
        valfmt="%g"
    )
    slider_axes.append(ax_slider)
    sliders.append(slider)

# Update function
def update(val):
    current_params = {name: slider.val for name, slider in zip(params.keys(), sliders)}
    sol = odeint(base_model, y0, t_fine, args=tuple(current_params.values()))
    line.set_ydata(sol[:, 0])
    fig.canvas.draw_idle()

# Register update function
for slider in sliders:
    slider.on_changed(update)

# Reset button
reset_ax = plt.axes([0.8, 0.02, 0.1, 0.04])
reset_button = Button(reset_ax, 'Reset', color='lightgoldenrodyellow')

def reset(event):
    for name, slider in zip(params.keys(), sliders):
        slider.set_val(params[name])  # Reset to optimized values

reset_button.on_clicked(reset)

# Plot formatting
ax.set_title("Interactive Tumor Growth Model - Optimized Starting Fit", fontsize=14)
ax.set_ylabel('Tumor Volume (mm³)', fontsize=12)
ax.set_xlabel('Time (days)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(['Model Fit', 'Experimental Data'], loc='upper left')

plt.show()