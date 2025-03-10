import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Oscillator:
    def __init__(self, theta, omega):
        """Initialize an oscillator with phase and natural frequency"""
        self.theta = theta  # Phase
        self.omega = omega  # Natural frequency

    def update(self, neighbors, K, dt):
        """Update the phase based on neighboring oscillators"""
        coupling = np.sum(np.sin([n.theta - self.theta for n in neighbors]))
        self.theta += (self.omega + K * coupling / len(neighbors)) * dt
        self.theta %= 2 * np.pi  # Keep phase within 0 to 2π


class Synchronizer:
    def __init__(self, num_oscillators=20, K=2.0, dt=0.005):
        """Initialize the synchronizer model"""
        self.K = K  # Coupling strength
        self.dt = dt  # Time step
        self.oscillators = [Oscillator(np.random.rand() * 2 * np.pi, np.random.normal(1, 0.1))
                            for _ in range(num_oscillators)]

    def update(self):
        """Update the phase of all oscillators"""
        for osc in self.oscillators:
            osc.update(self.oscillators, self.K, self.dt)

    def get_phases(self):
        """Get the current phase of all oscillators"""
        return np.array([osc.theta for osc in self.oscillators])


# Coupling strengths to compare
K_values = [0.25, 0.5, 0.75, 1.0]

# Create figure with three subplots
fig, axes = plt.subplots(1, 4, subplot_kw={'projection': 'polar'}, figsize=(12, 4))

# Initialize synchronizer models for each subplot
sync_models = [Synchronizer(num_oscillators=20, K=K, dt=0.005) for K in K_values]

# Scatter plots for each subplot
scatters = [ax.scatter([], [], c=[], cmap='hsv', s=80) for ax in axes]

# Set titles and remove axis labels
for ax, K in zip(axes, K_values):
    ax.set_title(f"K = {K}")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Initialization function
def init():
    for sc in scatters:
        sc.set_offsets(np.zeros((50, 2)))
        sc.set_array(np.zeros(50))
    return scatters

# Update function for animation
def update(frame):
    for i, (sync_model, sc) in enumerate(zip(sync_models, scatters)):
        sync_model.update()  # Update oscillator states
        phases = sync_model.get_phases()  # Get current phases
        sc.set_offsets(np.column_stack((phases, np.ones(len(phases)))))  # Convert to polar coordinates
        sc.set_array(phases)  # Colour based on phase
    return scatters

# Animation settings
# Update dt for quicker updates
sync_models = [Synchronizer(num_oscillators=20, K=K, dt=0.05) for K in K_values]

# Animation settings (faster and smoother)
num_frames = 800  # More frames for smooth animation
interval_ms = 0.01    # Reduce interval for faster rendering

ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, interval=interval_ms, blit=True)
# ani.save(f'111kuramoto_sync_k={K_values}.gif', writer='imagemagick',fps=120)
plt.show()
