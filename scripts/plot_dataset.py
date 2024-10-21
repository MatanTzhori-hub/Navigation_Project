import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_trajectory(v_optimal, phi_optimal, initial_state, T, dt, L):
        x, y, theta = initial_state
        trajectory_x, trajectory_y, trajectory_theta = [x], [y], [theta]
        for _ in np.arange(0, T, dt):
            x += v_optimal * np.cos(theta) * dt
            y += v_optimal * np.sin(theta) * dt
            theta += (v_optimal / L) * np.tan(phi_optimal) * dt
            trajectory_x.append(x)
            trajectory_y.append(y)
            trajectory_theta.append(theta)  
        return (trajectory_x, trajectory_y, trajectory_theta)


def plot_trajectory(v_optimal, phi_optimal, initial_state, T, dt, L, color):
    trajectory_x, trajectory_y, trajectory_theta = get_trajectory(v_optimal, phi_optimal, initial_state, T, dt, L)
    
    plt.quiver(trajectory_x[::10], trajectory_y[::10], 
                np.cos(trajectory_theta[::10]), np.sin(trajectory_theta[::10]), 
                scale=100, width=0.002, color=color, label='_Hidden label')
    plt.plot(trajectory_x, trajectory_y, color=color)

# Initialize starting state
start = [0, 0, 0]
filename = 'dataset/BigDS_Filttered.csv'
data = np.genfromtxt(filename, delimiter=',', skip_header=1)

rows = data.shape[0]
n_samples = 50

# Generate random sample indices
sample_indices = np.random.choice(rows, size=n_samples, replace=False)
sampled_rows = data[sample_indices]

# Create a colormap
colors = cm.viridis(np.linspace(0, 1, n_samples))

# Loop through sampled rows to plot trajectories
for i in range(n_samples):
    start_i = start + np.random.uniform(0, 20, size=3)
    sample = sampled_rows[i]
    v = sample[3]
    delta = sample[4]
    T = sample[5]
    
    # Plot with a different color for each trajectory
    plot_trajectory(v, delta, start, T, 0.01, 2, color=colors[i])

# Show the plot
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trajectory')
plt.grid(True)
plt.show()
