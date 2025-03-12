import matplotlib
matplotlib.use('TkAgg')  # Forces Matplotlib to use an external window

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit

# Standard Kuramoto Model
def kuramoto(theta, t, K, omega):
    N = len(theta)
    dtheta_dt = np.zeros(N)
    for i in range(N):
        dtheta_dt[i] = omega[i] + (K / N) * np.sum(np.sin(theta - theta[i]))
    return dtheta_dt

# Parameters
N = 8  # Total oscillators (4 per group)
K = 1.22  # Coupling strength

# Assign natural frequencies in a **bimodal distribution**
base_frequency = 0
frequency_shift = 1
natural_frequencies = np.array(
    [base_frequency - frequency_shift] * (N // 2) +
    [base_frequency + frequency_shift] * (N // 2)
)

# Generate stochastic initial phases **centered at π and 2π ± 0.75π**
stochastic_variation = np.random.uniform(-0.75 * np.pi, 0.75 * np.pi, N)
theta0 = np.zeros(N)
theta0[:N // 2] = np.pi + stochastic_variation[:N // 2]  # Group 1 around π
theta0[N // 2:] = 2 * np.pi + stochastic_variation[N // 2:]  # Group 2 around 2π

# Time step parameters
dt = 0.075  # Small time step for precision
max_time = 30  # Upper limit to prevent infinite loops

# Initialize time and phase evolution tracking
time = np.arange(0, max_time, dt)
theta_t = odeint(kuramoto, theta0, time, args=(K, natural_frequencies))

# **Map Phases to a Smooth Projection (X-axis Representation)**
phi_t = np.arctan2(np.sin(theta_t), np.cos(theta_t))  # Ensures [-π, π] range
x_projection = np.cos(theta_t)  # X-coordinate projection to smooth wave

# **Detect First Entry Into Antiphase (Phase Difference Near π)**
antiphase_time = None
threshold = 0.1 * np.pi  # Allow 10% variation around π
for t_idx in range(len(time) - 1):  # Check phase difference between successive times
    phase_diffs = np.abs(phi_t[t_idx, :N//2] - phi_t[t_idx, N//2:]) % (2 * np.pi)
    mean_phase_diff = np.mean(phase_diffs)  # Average difference between groups

    # Check for crossing into antiphase (mean phase difference ≈ π)
    if np.abs(mean_phase_diff - np.pi) < threshold and antiphase_time is None:
        antiphase_time = time[t_idx]  # Mark first occurrence of antiphase

# Convert to x, y coordinates for visualization on a unit circle
x = np.cos(theta_t)
y = np.sin(theta_t)
# -------------------------------
# Plot 1: Animation of Oscillators on Unit Circle
# + a Node for the Average Position
# -------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Kuramoto Model - Oscillator Motion (With Average Position)")

# Draw a unit circle
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='dashed')
ax.add_patch(circle)

# Initialize oscillator points
oscillators, = ax.plot([], [], 'ro', markersize=8, label="Oscillators")

# Initialize connecting lines between oscillators and origin
lines = [ax.plot([], [], 'k-', alpha=0.5)[0] for _ in range(N)]

# --- NEW: Add an extra marker/line for the average position ---
avg_oscillator, = ax.plot([], [], 'bo', markersize=10, label="Average Position")


# NEW: Initialize a line to show the path of the average point
avg_path, = ax.plot([], [], 'm-', lw=2, label="Average Path")
avg_path_x = []
avg_path_y = []

plt.legend()
plt.ion()  # Turn interactive mode on

# Manual Animation Loop
for i in range(len(time)):
    # Update oscillator positions
    oscillators.set_data(x[i], y[i])

    # Update lines from origin to each oscillator
    for j in range(N):
        lines[j].set_data([0, x[i, j]], [0, y[i, j]])

    # Compute the average (mean) x, y
    avg_x = np.mean(x[i])
    avg_y = np.mean(y[i])

    # Update the average oscillator marker and line
    # Wrap the single scalar in a list or array
    avg_oscillator.set_data([avg_x], [avg_y])


    # Update the average path
    avg_path_x.append(avg_x)
    avg_path_y.append(avg_y)
    avg_path.set_data(avg_path_x, avg_path_y)

    plt.draw()
    plt.pause(0.01)  # Adjust for smoother/faster animation

plt.ioff()  # Turn interactive mode off
plt.show()

# ----------------------------------------------------
# Now the rest of your analysis with theta_dot, etc.
# ----------------------------------------------------







# Compute theta_dot (angular velocity) using finite differences
theta_dot = np.abs(np.gradient(theta_t, axis=0) / dt)

# Plot theta_dot over time
plt.figure(figsize=(10, 5))
for i in range(N):
    plt.plot(time, theta_dot[:, i], label=f'Oscillator {i+1}')

plt.xlabel("Time")
plt.ylabel("Theta Dot (Instantaneous Frequency)")
plt.title("Instantaneous Frequency Evolution of Oscillators")
plt.legend()
plt.grid()
plt.show()

# Define a threshold for frequency synchronization (small variation in theta_dot values)
sync_threshold = 0.0005  # Adjust this value for tighter/looser synchronization detection

# Find the first time step where all theta_dot values are within the threshold range
sync_time = None
for t_idx in range(len(time)):
    max_theta_dot = np.max(theta_dot[t_idx])
    min_theta_dot = np.min(theta_dot[t_idx])
    if (max_theta_dot - min_theta_dot) < sync_threshold:
        sync_time = time[t_idx]
        break  # Stop at the first occurrence of synchronization

# Compute the x-projection again for plotting
x_projection = np.cos(theta_t)

# Plot the "Smooth Projection of Circular Motion" with the synchronization point marked
plt.figure(figsize=(10, 5))
for i in range(N):
    plt.plot(time, x_projection[:, i], label=f'Oscillator {i+1}')  # X Projection for smoothness

# Highlight the detected synchronization point with a vertical line
if sync_time is not None:
    plt.axvline(x=sync_time, color='blue', linestyle='--', label=f"AntiPhase Detected at t={sync_time:.2f}s")

plt.xlabel("Time")
plt.ylabel("Current Phase of Node")
plt.yticks([-1, -0.5, 0, 0.5, 1], [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
plt.title("Smooth Projection of Circular Motion with Synchronization Point at 0.05% Tolerance")
plt.legend(loc="upper right")
plt.grid()
plt.show()



# -----------------------------
# Common simulation parameters
# -----------------------------
base_frequency = 0
frequency_shift = 1
dt = 0.075      # time step
max_time = 15   # maximum simulation time
time = np.arange(0, max_time, dt)
sync_threshold = 0.0005  # threshold for theta_dot synchronization detection
num_runs = 100          # number of runs per parameter value

# ==============================================================================
# Section 1: Varying Coupling Strength (K) using theta_dot detection method
# ==============================================================================
fixed_total_nodes = 8
group_size_fixed = fixed_total_nodes // 2

# Define a range of coupling strengths
k_values = np.linspace(0.5, 4.0, 100)
avg_sync_times_k = []

# Determine the index corresponding to 6 seconds (approx).
index_6sec = int(13.0 / dt)
# Set the threshold for the overall mean theta_dot at 6 seconds.
threshold_overall = 0.01

for current_K in k_values:
    run_sync_times = []
    theta_dot_at_6_all_runs = []
    for run in range(num_runs):
        # Generate randomized initial conditions:
        stochastic_variation = np.random.uniform(-0.75 * np.pi, 0.75 * np.pi, fixed_total_nodes)
        theta0 = np.zeros(fixed_total_nodes)
        theta0[:group_size_fixed] = np.pi + stochastic_variation[:group_size_fixed]
        theta0[group_size_fixed:] = 2 * np.pi + stochastic_variation[group_size_fixed:]

        # Bimodal natural frequencies for the two groups
        natural_frequencies = np.array([base_frequency - frequency_shift] * group_size_fixed +
                                       [base_frequency + frequency_shift] * group_size_fixed)

        # Integrate Kuramoto ODE
        theta_t = odeint(kuramoto, theta0, time, args=(current_K, natural_frequencies))
        theta_dot = np.abs(np.gradient(theta_t, axis=0) / dt)

        # Detect the first time when the spread (max-min) of theta_dot < sync_threshold
        sync_time = np.nan
        for t_idx in range(len(time)):
            if np.max(theta_dot[t_idx]) - np.min(theta_dot[t_idx]) < sync_threshold:
                sync_time = time[t_idx]
                break

        run_sync_times.append(sync_time)
        # Record the average theta_dot (across nodes) at ~6 seconds
        theta_dot_at_6_all_runs.append(np.mean(theta_dot[index_6sec]))

    # Now compute the overall mean theta_dot at 6 seconds across all runs.
    overall_mean_theta_dot = np.mean(theta_dot_at_6_all_runs)
    # If the overall mean is below threshold_overall, discard by setting sync time to NaN.
    if overall_mean_theta_dot < threshold_overall:
        avg_sync_times_k.append(np.nan)
    else:
        # Otherwise, average only the valid runs (non-NaN sync times)
        valid = np.array(run_sync_times)[~np.isnan(run_sync_times)]
        if valid.size > 0:
            avg_sync_times_k.append(valid.mean())
        else:
            avg_sync_times_k.append(np.nan)

# # # # # # # # # # # # # # # # # # # # # # #
# Multi-degree polynomial fit and plotting
# # # # # # # # # # # # # # # # # # # # # # #

# Convert to arrays for convenience
k_values = np.array(k_values)
avg_sync_times_k = np.array(avg_sync_times_k)

# Filter out NaN values so polyfit doesn't crash
valid_mask = ~np.isnan(avg_sync_times_k)
k_valid = k_values[valid_mask]
sync_valid = avg_sync_times_k[valid_mask]

# Create a dense array of K for smooth polynomial curves
k_dense = np.linspace(min(k_valid), max(k_valid), 300)

# We'll try multiple degrees
degrees_to_try = [4]
fits_info = []

for deg in degrees_to_try:
    coeffs = np.polyfit(k_valid, sync_valid, deg=deg)
    poly_func = np.poly1d(coeffs)

    # Compute predictions on valid data to get R^2
    fitted_values = poly_func(k_valid)
    residuals = sync_valid - fitted_values
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((sync_valid - np.mean(sync_valid)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    fits_info.append((deg, coeffs, r2, poly_func))

# Plot the data and all polynomial fits
plt.figure(figsize=(8, 5))
plt.scatter(k_values, avg_sync_times_k, label="Data", color="blue")

colors = ["red", "green", "magenta", "orange"]
for i, (deg, coeffs, r2, poly_func) in enumerate(fits_info):
    plt.plot(k_dense, poly_func(k_dense), color=colors[i], linestyle="--",
             label=f"Degree={deg}, R²={r2:.2f}")

plt.xlabel("Coupling Strength (K)")
plt.ylabel("Antiphase Synchronization Time (s)")
plt.title("Antiphase Sync Time vs. Coupling Strength (with Quartic Fit)")
plt.legend()
plt.grid(True)
plt.show()


# -----------------------------
# Section 2: Varying Group Size using theta_dot detection method (K fixed at 1.22)
# -----------------------------
K = 2.22
group_sizes = np.arange(2, 101)  # nodes per group (so total nodes = 2 * group_size)
avg_sync_times_group = []

index_6sec = int(13.0 / dt)  # index corresponding to ~25 seconds (or your desired index)
threshold_overall = 1e-3  # threshold for the overall mean theta_dot at 6 sec

for group_size in group_sizes:
    N = 2 * group_size  # total nodes
    natural_frequencies = np.array(
        [base_frequency - frequency_shift] * group_size +
        [base_frequency + frequency_shift] * group_size
    )
    run_sync_times = []
    theta_dot_at_6_all_runs = []  # store mean theta_dot (across nodes) at index_6sec for each run

    for run in range(num_runs):
        # Generate randomized initial conditions
        stochastic_variation = np.random.uniform(-0.75 * np.pi, 0.75 * np.pi, N)
        theta0 = np.zeros(N)
        theta0[:group_size] = np.pi + stochastic_variation[:group_size]   # Group 1
        theta0[group_size:] = 2 * np.pi + stochastic_variation[group_size:] # Group 2

        # Simulate the Kuramoto model (K fixed at 1.22)
        theta_t = odeint(kuramoto, theta0, time, args=(1.22, natural_frequencies))
        # Compute theta_dot using finite differences
        theta_dot = np.abs(np.gradient(theta_t, axis=0) / dt)

        # Determine the first time when the spread (max - min) of theta_dot falls below sync_threshold
        sync_time = np.nan
        for t_idx in range(len(time)):
            if np.max(theta_dot[t_idx]) - np.min(theta_dot[t_idx]) < sync_threshold:
                sync_time = time[t_idx]
                break
        run_sync_times.append(sync_time)

        # Record the mean theta_dot across nodes at the given index (e.g., ~25 seconds)
        theta_dot_at_6_all_runs.append(np.mean(theta_dot[index_6sec]))

    # Compute the overall mean theta_dot at the selected index across all runs.
    overall_mean_theta_dot = np.mean(theta_dot_at_6_all_runs)
    if overall_mean_theta_dot < threshold_overall:
        # If overall mean theta_dot is below the threshold, mark this parameter value as invalid.
        avg_sync_times_group.append(np.nan)
    else:
        # Otherwise, average only the valid (non-NaN) sync times.
        valid = np.array(run_sync_times)[~np.isnan(run_sync_times)]
        if valid.size > 0:
            avg_sync_times_group.append(valid.mean())
        else:
            avg_sync_times_group.append(np.nan)

# -----------------------------
# Fitting Logarithmic and Exponential Models
# -----------------------------
# Convert to arrays for convenience.
group_sizes_arr = np.array(group_sizes)
avg_sync_times_group_arr = np.array(avg_sync_times_group)

# Filter out NaN values.
valid_mask = ~np.isnan(avg_sync_times_group_arr)
group_valid = group_sizes_arr[valid_mask]
sync_valid = avg_sync_times_group_arr[valid_mask]

# Define logarithmic and exponential fit functions
def log_fit(x, a, b):
    return a * np.log(x) + b



# Fit logarithmic function
log_params, _ = curve_fit(log_fit, group_valid, sync_valid)
log_fit_values = log_fit(group_valid, *log_params)



# Compute R-squared values
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

r2_log = r_squared(sync_valid, log_fit_values)


# Create a dense x-axis array for smooth plotting
x_dense = np.linspace(min(group_valid), max(group_valid), 300)

# -----------------------------
# Plotting Polynomial, Log, and Exp Fits
# -----------------------------
plt.figure(figsize=(8, 5))
plt.scatter(group_sizes_arr, avg_sync_times_group_arr, label="Data", color="blue")

# Quartic Fit (Polynomial)
degrees_to_try = []
colors = ["red"]
for i, deg in enumerate(degrees_to_try):
    coeffs = np.polyfit(group_valid, sync_valid, deg=deg)
    poly_func = np.poly1d(coeffs)
    plt.plot(x_dense, poly_func(x_dense), color=colors[i], linestyle="--",
             label=f"Quartic Fit (Degree {deg})")

# Logarithmic Fit
plt.plot(x_dense, log_fit(x_dense, *log_params), 'g--', label=f"Log Fit ($R^2$={r2_log:.2f})")

# Exponential Fit


plt.xlabel("Nodes per Group")
plt.ylabel("Antiphase Synchronization Time (s)")
plt.title("Antiphase Sync Time vs. Nodes per Group (with log Fit)")
plt.legend()
plt.grid()
plt.show()


# -----------------------------
# Configurations: Group 1 always has 1 oscillator, Group 2 varies from 1 to 10
# -----------------------------
configurations = range(1, 11)  # Group 2 oscillator count: 1, 2, ..., 10
n_configs = len(configurations)

# Create subplots in a 2 x 5 grid to display all animations simultaneously
nrows, ncols = 2, 5
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 6))
axes = axes.flatten()

# We'll store simulation data and the associated plot objects for each configuration.
sim_data = []  # list of dictionaries

for idx, group2_n in enumerate(configurations):
    group1_n = 1
    total_n = group1_n + group2_n

    # Define natural frequencies for a bimodal distribution:
    # Group 1: lower frequency, Group 2: higher frequency
    natural_frequencies = np.array([base_frequency - frequency_shift] * group1_n +
                                   [base_frequency + frequency_shift] * group2_n)

    # Generate stochastic initial phases:
    # Group 1 centered at π; Group 2 centered at 2π (with added random variation)
    stochastic_variation = np.random.uniform(-0.75 * np.pi, 0.75 * np.pi, total_n)
    theta0 = np.zeros(total_n)
    theta0[:group1_n] = np.pi + stochastic_variation[:group1_n]
    theta0[group1_n:] = 2 * np.pi + stochastic_variation[group1_n:]

    # Simulate the Kuramoto model for this configuration
    theta_t = odeint(kuramoto, theta0, time, args=(K, natural_frequencies))
    # Compute x,y positions on the unit circle
    x = np.cos(theta_t)
    y = np.sin(theta_t)

    # Set up the corresponding subplot
    ax = axes[idx]
    ax.cla()
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"1 vs {group2_n} Oscillators")

    # Draw a unit circle for reference
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='dashed')
    ax.add_patch(circle)

    # Create plot objects for oscillators and their connecting lines
    osc_marker, = ax.plot([], [], 'ro', markersize=6)  # oscillators (red dots)
    lines = []
    for j in range(total_n):
        line, = ax.plot([], [], 'k-', alpha=0.5)  # lines from origin to each oscillator
        lines.append(line)

    # Create plot objects for the average position:
    avg_marker, = ax.plot([], [], 'bo', markersize=8)  # average position (blue dot)
    avg_vector, = ax.plot([], [], 'b--', alpha=0.5)  # line from origin to average
    avg_path, = ax.plot([], [], 'm-', lw=1)  # path of the average (magenta line)

    # Save simulation data and plot objects in a dictionary
    sim_data.append({
        'total_n': total_n,
        'x': x,
        'y': y,
        'ax': ax,
        'osc_marker': osc_marker,
        'lines': lines,
        'avg_marker': avg_marker,
        'avg_vector': avg_vector,
        'avg_path': avg_path,
        'avg_path_x': [],
        'avg_path_y': []
    })

# -----------------------------
# Animate all subplots simultaneously
# -----------------------------
plt.ion()  # interactive mode on

for frame in range(len(time)):
    for data in sim_data:
        x_data = data['x'][frame]  # positions for all oscillators in this configuration
        y_data = data['y'][frame]
        # Update oscillator markers
        data['osc_marker'].set_data(x_data, y_data)
        # Update lines from origin to each oscillator
        for j in range(data['total_n']):
            data['lines'][j].set_data([0, x_data[j]], [0, y_data[j]])
        # Compute the average position at this frame
        avg_x = np.mean(x_data)
        avg_y = np.mean(y_data)
        data['avg_marker'].set_data([avg_x], [avg_y])
        data['avg_vector'].set_data([0, avg_x], [0, avg_y])
        # Append the average position to the path and update the path line
        data['avg_path_x'].append(avg_x)
        data['avg_path_y'].append(avg_y)
        data['avg_path'].set_data(data['avg_path_x'], data['avg_path_y'])
    plt.pause(0.01)

plt.ioff()
plt.show()

# Create 100 subplots in a 10x10 grid
nrows, ncols = 10, 10
fig, axes = plt.subplots(nrows, ncols, figsize=(20, 20))
axes = axes.flatten()

# List to store simulation data for each configuration
sim_data = []
config_index = 0

# Loop over group sizes for both groups: group1 and group2 vary from 1 to 10
for group1_n in range(1, 11):
    for group2_n in range(1, 11):
        total_n = group1_n + group2_n

        # Define natural frequencies:
        # Group 1 gets (base_frequency - frequency_shift) and Group 2 gets (base_frequency + frequency_shift)
        natural_frequencies = np.array(
            [base_frequency - frequency_shift] * group1_n +
            [base_frequency + frequency_shift] * group2_n
        )

        # Generate randomized initial phases:
        # Group 1 centered at π and Group 2 centered at 2π, with added stochastic variation.
        stochastic_variation = np.random.uniform(-0.75 * np.pi, 0.75 * np.pi, total_n)
        theta0 = np.zeros(total_n)
        theta0[:group1_n] = np.pi + stochastic_variation[:group1_n]
        theta0[group1_n:] = 2 * np.pi + stochastic_variation[group1_n:]

        # Simulate the Kuramoto model for this configuration
        theta_t = odeint(kuramoto, theta0, time, args=(K, natural_frequencies))
        x = np.cos(theta_t)  # x-coordinates on unit circle
        y = np.sin(theta_t)  # y-coordinates on unit circle

        # Set up the subplot
        ax = axes[config_index]
        ax.cla()
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{group1_n} vs {group2_n}")

        # Draw a unit circle for reference
        circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='dashed')
        ax.add_patch(circle)

        # Create plot objects for oscillators and connecting lines
        osc_marker, = ax.plot([], [], 'ro', markersize=4)  # Oscillator markers
        lines = []
        for j in range(total_n):
            line, = ax.plot([], [], 'k-', alpha=0.5)  # Line from origin to oscillator j
            lines.append(line)

        # Create plot objects for the average position:
        avg_marker, = ax.plot([], [], 'bo', markersize=5)  # Average position (blue dot)
        avg_vector, = ax.plot([], [], 'b--', alpha=0.5)  # Line from origin to average position
        avg_path, = ax.plot([], [], 'm-', lw=1)  # Path of the average position

        # Save simulation data and associated plot objects in a dictionary
        sim_data.append({
            'total_n': total_n,
            'x': x,
            'y': y,
            'ax': ax,
            'osc_marker': osc_marker,
            'lines': lines,
            'avg_marker': avg_marker,
            'avg_vector': avg_vector,
            'avg_path': avg_path,
            'avg_path_x': [],
            'avg_path_y': []
        })
        config_index += 1

plt.tight_layout()
plt.ion()  # Turn interactive mode on

# Animate all subplots simultaneously
for frame in range(len(time)):
    for data in sim_data:
        x_data = data['x'][frame]
        y_data = data['y'][frame]
        # Update oscillator positions
        data['osc_marker'].set_data(x_data, y_data)
        # Update lines connecting each oscillator to the origin
        for j in range(data['total_n']):
            data['lines'][j].set_data([0, x_data[j]], [0, y_data[j]])
        # Compute and update the average position of all oscillators in this configuration
        avg_x = np.mean(x_data)
        avg_y = np.mean(y_data)
        data['avg_marker'].set_data([avg_x], [avg_y])
        data['avg_vector'].set_data([0, avg_x], [0, avg_y])
        # Append the current average position to the path and update the path plot
        data['avg_path_x'].append(avg_x)
        data['avg_path_y'].append(avg_y)
        data['avg_path'].set_data(data['avg_path_x'], data['avg_path_y'])
    plt.pause(0.01)

plt.ioff()  # Turn interactive mode off
plt.show()

threshold_distance = 0.005
# We'll require the conditions to hold for at least 1 second.
required_frames = int(np.ceil(1.0 / dt))  # e.g. ~14 frames for dt=0.075

# -----------------------------
# Define group sizes:
# Group 1 sizes: {4, 8, 12, 16}
# For each Group 1, Group 2 = Group 1 + offset, offset in {-2, -1, 0, 1, 2}.
# Total configurations = 4 x 5 = 20.
# -----------------------------
group1_sizes = [4, 8, 12, 16]
offsets = [-2, -1, 0, 1, 2]
n_configs = len(group1_sizes) * len(offsets)

# Create a 4x5 grid of subplots with adjusted figure size and spacing.
fig, axes = plt.subplots(4, 5, figsize=(14, 10))
axes = axes.flatten()
plt.subplots_adjust(left=0.06, right=0.96, bottom=0.06, top=0.94, wspace=0.25, hspace=0.45)

# List to store simulation data for each configuration.
sim_data = []
config_index = 0

for group1_n in group1_sizes:
    for offset in offsets:
        group2_n = group1_n + offset
        total_n = group1_n + group2_n

        # Natural frequencies: Group 1 gets (base_frequency - frequency_shift),
        # Group 2 gets (base_frequency + frequency_shift).
        natural_frequencies = np.array(
            [base_frequency - frequency_shift] * group1_n +
            [base_frequency + frequency_shift] * group2_n
        )

        # Generate stochastic initial phases:
        # Group 1 centered at π, Group 2 centered at 2π (with random variation).
        stochastic_variation = np.random.uniform(-0.75 * np.pi, 0.75 * np.pi, total_n)
        theta0 = np.zeros(total_n)
        theta0[:group1_n] = np.pi + stochastic_variation[:group1_n]
        theta0[group1_n:] = 2 * np.pi + stochastic_variation[group1_n:]

        # Simulate the Kuramoto model for this configuration.
        theta_t = odeint(kuramoto, theta0, time, args=(K, natural_frequencies))
        x = np.cos(theta_t)  # x-coordinates on unit circle
        y = np.sin(theta_t)  # y-coordinates on unit circle
        # Precompute theta_dot using finite differences.
        theta_dot = np.abs(np.gradient(theta_t, axis=0) / dt)

        # Set up the corresponding subplot.
        ax = axes[config_index]
        ax.cla()
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"G1: {group1_n} vs G2: {group2_n}", fontsize=9)

        # Draw a unit circle for reference.
        circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='dashed')
        ax.add_patch(circle)

        # Create plot objects for oscillators and connecting lines.
        osc_marker, = ax.plot([], [], 'ro', markersize=5)  # Red dot for oscillators
        lines = []
        for j in range(total_n):
            line, = ax.plot([], [], 'k-', alpha=0.5)  # Line from origin to oscillator
            lines.append(line)

        # Create plot objects for the average position.
        avg_marker, = ax.plot([], [], 'bo', markersize=6)  # Blue dot for average position
        avg_vector, = ax.plot([], [], 'b--', alpha=0.5)  # Dashed line from origin to average
        avg_path, = ax.plot([], [], 'm-', lw=1)  # Magenta line for average path

        # Add new keys for antiphase lock.
        sim_data.append({
            'group1_n': group1_n,
            'group2_n': group2_n,
            'total_n': total_n,
            'theta_t': theta_t,  # full simulation data
            'theta_dot': theta_dot,  # precomputed theta_dot
            'x': x,
            'y': y,
            'ax': ax,
            'osc_marker': osc_marker,
            'lines': lines,
            'avg_marker': avg_marker,
            'avg_vector': avg_vector,
            'avg_path': avg_path,
            'avg_path_x': [],
            'avg_path_y': [],
            'antiphase_counter': 0,  # counter for consecutive frames holding conditions
            'locked': False  # flag if antiphase is permanently detected
        })
        config_index += 1

plt.tight_layout()
plt.ion()  # Turn interactive mode on

# Animate all subplots simultaneously.
for frame in range(len(time)):
    for data in sim_data:
        x_data = data['x'][frame]  # Oscillator positions (x) at current frame.
        y_data = data['y'][frame]  # Oscillator positions (y) at current frame.
        # Update oscillator markers.
        data['osc_marker'].set_data(x_data, y_data)
        # Update lines from origin to each oscillator.
        for j in range(data['total_n']):
            data['lines'][j].set_data([0, x_data[j]], [0, y_data[j]])
        # Compute and update the overall average position.
        avg_x = np.mean(x_data)
        avg_y = np.mean(y_data)
        data['avg_marker'].set_data([avg_x], [avg_y])
        data['avg_vector'].set_data([0, avg_x], [0, avg_y])
        data['avg_path_x'].append(avg_x)
        data['avg_path_y'].append(avg_y)
        data['avg_path'].set_data(data['avg_path_x'], data['avg_path_y'])

        # --- Theta_dot Based Antiphase Detection ---
        # Condition 1: Spread of theta_dot is below sync_threshold.
        current_theta_dot = data['theta_dot'][frame]
        cond1 = (np.max(current_theta_dot) - np.min(current_theta_dot)) < sync_threshold

        # Condition 2: Overall average is nearly equidistant from each group's average.
        g1 = data['group1_n']
        g2 = data['group2_n']
        group1_avg = np.array([np.mean(x_data[:g1]), np.mean(y_data[:g1])])
        group2_avg = np.array([np.mean(x_data[g1:g1 + g2]), np.mean(y_data[g1:g1 + g2])])
        overall_avg = np.array([avg_x, avg_y])
        d1 = np.linalg.norm(overall_avg - group1_avg)
        d2 = np.linalg.norm(overall_avg - group2_avg)
        # Check if d2 is within 98% to 102% of d1
        cond2 = (d2 >= 0.98 * d1) and (d2 <= 1.02 * d1)

        # Only update if not locked already.
        if not data['locked']:
            if cond1 and cond2:
                data['antiphase_counter'] += 1
            else:
                data['antiphase_counter'] = 0
            # If conditions have held for at least the required frames, lock in antiphase.
            if data['antiphase_counter'] >= required_frames:
                data['locked'] = True
                data['ax'].set_facecolor('limegreen')
            else:
                data['ax'].set_facecolor('white')
        else:
            data['ax'].set_facecolor('limegreen')
    plt.pause(0.01)

plt.ioff()  # Turn interactive mode off
plt.show()


# -----------------------------
# Simulation parameters
# -----------------------------
group1_n = 1000  # 20 nodes in group 1
group2_n = 1  # 1 node in group 2
total_n = group1_n + group2_n  # total of 21 oscillators

K = 1.22  # Coupling strength

base_frequency = 0
frequency_shift = 1

# Define natural frequencies:
# Group 1 gets a lower frequency; Group 2 gets a higher frequency.
natural_frequencies = np.array(
    [base_frequency - frequency_shift] * group1_n +
    [base_frequency + frequency_shift] * group2_n
)

# Generate stochastic initial phases:
# Group 1 centered at π; Group 2 centered at 2π, with added random variation.
stochastic_variation = np.random.uniform(-0.75 * np.pi, 0.75 * np.pi, total_n)
theta0 = np.zeros(total_n)
theta0[:group1_n] = np.pi + stochastic_variation[:group1_n]  # Group 1 around π
theta0[group1_n:] = 2 * np.pi + stochastic_variation[group1_n:]  # Group 2 around 2π

dt = 0.075  # Time step
max_time = 30  # Total simulation time (seconds)
time = np.arange(0, max_time, dt)

# Solve the Kuramoto model ODE
theta_t = odeint(kuramoto, theta0, time, args=(K, natural_frequencies))

# Map the phases to x,y positions on the unit circle.
x = np.cos(theta_t)
y = np.sin(theta_t)

# -----------------------------
# Setup the plot for animation
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 12))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Kuramoto Model: 1000 vs 1 Oscillators", fontsize=27)

# Draw a dashed unit circle for reference.
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='dashed')
ax.add_patch(circle)

# Create plot objects for the oscillators and their connecting lines.
oscillators, = ax.plot([], [], 'ro', markersize=5, label="Oscillators")
lines = [ax.plot([], [], 'k-', alpha=0.5)[0] for _ in range(total_n)]

# Create plot objects for the average position and its path.
avg_marker, = ax.plot([], [], 'bo', markersize=6, label="Average Position")
avg_vector, = ax.plot([], [], 'b--', alpha=0.5, label="Average Vector")
avg_path, = ax.plot([], [], 'm-', lw=1, label="Average Path")


# -----------------------------
# Animate the simulation with an extra tracer for Group 2 node
# -----------------------------
plt.ion()  # Turn interactive mode on
avg_path_x = []
avg_path_y = []

# Path for the singular node (Group 2)
group2_path_x = []
group2_path_y = []

# Create plot object for Group 2 tracer
group2_tracer, = ax.plot([], [], 'o', color='lime', markersize=6, label="Group 2 Tracer")

# Create plot object for Group 2 path
group2_path, = ax.plot([], [], 'lime', lw=1.5, label="Group 2 Path")  # Lime green path

save_times = [0, 3, 6, 9, 12, 20]

for frame in range(len(time)):
    # Update oscillator positions for this frame.
    oscillators.set_data(x[frame], y[frame])
    for j in range(total_n):
        lines[j].set_data([0, x[frame, j]], [0, y[frame, j]])

    # Compute and update the overall average position.
    avg_x = np.mean(x[frame])
    avg_y = np.mean(y[frame])
    avg_marker.set_data([avg_x], [avg_y])
    avg_vector.set_data([0, avg_x], [0, avg_y])

    # Update the average path.
    avg_path_x.append(avg_x)
    avg_path_y.append(avg_y)
    avg_path.set_data(avg_path_x, avg_path_y)

    # Update the Group 2 tracer
    group2_x = x[frame, -1]  # Last node is Group 2
    group2_y = y[frame, -1]
    group2_tracer.set_data([group2_x], [group2_y])

    # Track Group 2 path and update path plot
    group2_path_x.append(group2_x)
    group2_path_y.append(group2_y)
    group2_path.set_data(group2_path_x, group2_path_y)  # Update the lime green path

    plt.draw()
    plt.pause(0.01)

    # Save the plot at the specified times (with a tolerance of dt/2)
    if any(np.isclose(time[frame], t, atol=dt / 2) for t in save_times):
        plt.savefig(f"1vs1000_{int(time[frame])}s.png")

plt.ioff()  # Turn interactive mode off
plt.show()
