import numpy as np
import matplotlib.pyplot as plt
import scipy

Q, q = 1.6e-19, 1.6e-19 # mass of particle at origin and target particle respectively
const = 8.987e9 * Q * q # precomputes k*Q*q
m = 9.11e-31 # mass of particle

def func(t, y): # solver function
    return [
        y[1],
        (const / (m * y[0] ** 2)) + y[0]*y[3]**2,
        y[3],
        -(2 * y[1] * y[3]) / y[0]
    ]

def trajectory(b: float, d: float, v: float):

    t_span = [0, 1]
    
    y0 = [
        np.sqrt(b**2 + d**2),
        -v*np.cos(np.arctan(b/d)),
        np.arctan(b/d),
        (v*np.sin(np.arctan(b/d))) / np.sqrt(b**2 + d**2)
    ]

    def event(t, y): # terminates solver when particles distance to origin is same as at t=0
        return b**2 + d**2 - y[0]**2
    event.terminal = True
    event.direction = -1

    sol = scipy.integrate.solve_ivp(func, t_span, y0, method='DOP853', max_step=1e-6, events=(event), rtol=3e-7, atol=1e-12)
    return sol

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

b_values = np.linspace(-1e-9, 1e-9, 100)

for b_ in b_values:
    sol = trajectory(b_, 1e-5, 2189781)
    ax.plot(sol.y[2], sol.y[0], 'r-')

ax.set_rmax(5e-9)

plt.show()