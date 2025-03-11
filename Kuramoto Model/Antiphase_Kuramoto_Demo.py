import matplotlib

matplotlib.use('TkAgg')  # Forces Matplotlib to use an external window

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Standard Kuramoto Model
def kuramoto(theta, t, K, omega):
    N = len(theta)
    dtheta_dt = np.zeros(N)
    for i in range(N):
        dtheta_dt[i] = omega[i] + (K / N) * np.sum(np.sin(theta - theta[i]))
    return dtheta_dt


# Parameters
N = 10  # Total oscillators (4 per group)
K = 1.22  # Coupling strength

# Generate random natural frequencies for each group and adjust so the mean is exactly 1
group1_omega = np.random.uniform(0.98, 1.02, N // 2)
group2_omega = np.random.uniform(0.98, 1.02, N // 2)
group1_omega = group1_omega - np.mean(group1_omega) + 1
group2_omega = group2_omega - np.mean(group2_omega) + 1
natural_frequencies = np.concatenate([group1_omega, group2_omega])

# Initialize phases: group 1 at 0 radians and group 2 at π radians.
theta0 = np.zeros(N)
theta0[:N // 2] = 0  # Group 1 exactly at 0 radians
theta0[N // 2:] = np.pi  # Group 2 exactly at π radians

# Time step parameters
dt = 0.075  # Small time step for precision
max_time = 30  # Upper limit for the simulation time
time = np.arange(0, max_time, dt)

# Solve the differential equation
theta_t = odeint(kuramoto, theta0, time, args=(K, natural_frequencies))

# Convert phase to x,y coordinates on the unit circle
x = np.cos(theta_t)
y = np.sin(theta_t)

# Set up the plot for the animation
fig, ax = plt.subplots(figsize=(8, 12))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Oscillator Motion (With Average Position)", fontsize=27)

# Draw a unit circle for reference
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='dashed')
ax.add_patch(circle)

# Initialize oscillator points and connecting lines
oscillators, = ax.plot([], [], 'ro', markersize=8, label="Oscillators")
lines = [ax.plot([], [], 'k-', alpha=0.5)[0] for _ in range(N)]

# Marker and path for the average position of oscillators
avg_oscillator, = ax.plot([], [], 'bo', markersize=10, label="Average Position")
avg_path, = ax.plot([], [], 'm-', lw=2, label="Average Path")
avg_path_x = []
avg_path_y = []


plt.ion()  # Turn interactive mode on

# Define times at which to save the plot images
save_times = [0, 5, 10, 15, 20]

# Manual Animation Loop
for i in range(len(time)):
    # Update oscillator positions
    oscillators.set_data(x[i], y[i])

    # Update connecting lines from the origin to each oscillator
    for j in range(N):
        lines[j].set_data([0, x[i, j]], [0, y[i, j]])

    # Compute and update the average oscillator position
    avg_x = np.mean(x[i])
    avg_y = np.mean(y[i])
    avg_oscillator.set_data([avg_x], [avg_y])

    # Update the average path
    avg_path_x.append(avg_x)
    avg_path_y.append(avg_y)
    avg_path.set_data(avg_path_x, avg_path_y)

    plt.draw()
    plt.pause(0.01)

    # Save the plot at the specified times (with a tolerance of dt/2)
    if any(np.isclose(time[i], t, atol=dt / 2) for t in save_times):
        plt.savefig(f"kuramoto_{int(time[i])}s.png")

plt.ioff()  # Turn interactive mode off
plt.show()
