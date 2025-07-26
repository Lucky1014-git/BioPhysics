import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- 1. Time points ---
time_points = np.array([
    25, 26, 28, 31, 33, 35, 38, 40, 42, 45, 47, 49, 52, 54, 56, 59, 61, 63, 65
])

# --- 2. Raw vehicle control data (including silenced) ---
raw_data = [
    ["111", "115", "115", "148", "102"],
    ["120", "123", "123", "156", "110"],
    ["131", "132", "127", "168", "116"],
    ["141", "144", "136", "183", "126"],
    ["154", "154", "148", "202", "138"],
    ["172", "179", "166", "216", "175"],
    ["185", "195", "181", "237", "191"],
    ["196", "207", "195", "254", "207"],
    ["215", "219", "204", "269", "222"],
    ["359", "413", "310", "370", "285"],
    ["521", "447", "352", "447", "345"],
    ["546", "470", "371", "472", "364"],
    ["708", "599", "510", "661", "638"],
    ["759", "637", "538", "690", "674"],
    ["853", "993", "637", "783", "759"],
    ["1021", "", "759", "985", ""],
    ["", "", "933*", "1231*", ""],
    ["", "", "1001*", "", ""],
    ["", "", "1247*", "", ""]
]

# --- 3. Flatten all valid (time, value) pairs ---
all_time = []
all_values = []

for col in range(5):
    for i, row in enumerate(raw_data):
        if i >= len(time_points):  # skip if no time point
            continue
        val = row[col].strip()
        if val == "":
            continue
        val = val.replace("*", "")  # include silenced data
        all_time.append(time_points[i])
        all_values.append(float(val))

all_time = np.array(all_time)
all_values = np.array(all_values)

# --- 4. Define tumor growth model ---
def tumor_model(T, t, lam):
    return lam * T

# --- 5. Prediction function ---
def predict(lam, t_array, T0):
    return odeint(tumor_model, T0, t_array, args=(lam,)).flatten()

# --- 6. SSR function to minimize ---
def total_ssr(params):
    lam = params[0]
    T0 = params[1]
    predictions = predict(lam, all_time, T0)
    return np.sum((all_values - predictions) ** 2)

# --- 7. Minimize SSR ---
initial_guess = [0.1, 100]  # [lambda, T0]
bounds = [(1e-5, 1.0), (50, 200)]
result = minimize(total_ssr, initial_guess, bounds=bounds)
lam_best, T0_best = result.x

# --- 8. Predict using best-fit parameters ---
t_fit = np.linspace(min(all_time), max(all_time), 200)
T_fit = predict(lam_best, t_fit, T0_best)

# --- 9. Plot ---
plt.figure(figsize=(10, 6))
