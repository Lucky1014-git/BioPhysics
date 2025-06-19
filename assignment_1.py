import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

beta = 1e-5     # infection rate
delta = 4       # death rate of infected cells
p = 2e6         # virus production rate
c = 4           # virus clearance rate

T0 = 1
I0 = 0
V0 = 0.01
y0 = [T0, I0, V0]

t = np.linspace(0, 50, 1000)

def viral_model(y, t, beta, delta, p, c):
    T, I, V = y
    dTdt = -beta * T * V
    dIdt = beta * T * V - delta * I
    dVdt = p * I - c * V
    return [dTdt, dIdt, dVdt]


print(viral_model(y0, t, beta, delta, p, c))
solution = odeint(viral_model, y0, t, args=(beta, delta, p, c))
T, I, V = solution.T

plt.figure(figsize=(10, 6))
plt.plot(t, T, label='Target cells (T)')
plt.plot(t, I, label='Infected cells (I)')
plt.plot(t, V, label='Virus (V)')
plt.yscale('log')  # log scale for virus
plt.xlabel('Time (hours)')
plt.ylabel('Population')
plt.title('Viral Infection Model')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



