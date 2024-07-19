import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define parameters
L = 2.0  # Wheelbase
T = 1 #60  # Time horizon
dt = 0.01  # Time step

# Initial and goal states
initial_state = np.array([-5, 0, np.pi/2])  # [x, y, theta]
goal_state = np.array([5, 5, 0])  # [x_goal, y_goal, theta_goal]

# Define cost function and constraints
def cost_function(u, initial_state, goal_state):
    x, y, theta = initial_state
    x_goal, y_goal, theta_goal = goal_state
    v, phi = u[0], u[1]
    cost = 0
    for _ in np.arange(0, T, dt):
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += (v / L) * np.tan(phi) * dt
    cost += (x - x_goal)**2 + (y - y_goal)**2 + (theta - theta_goal)**2
    return cost

# Initial guess for control inputs (flattened array of shape (2*T,))
u0 = [5, 0]

# Optimize
result = minimize(cost_function, u0, args=(initial_state, goal_state), method='SLSQP', options={'maxiter': 1000})

# Extract optimal control inputs
optimal_u = result.x

# Extract v and phi for plotting
v_optimal, phi_optimal = optimal_u[0], optimal_u[1]

# Apply control inputs in a loop to get the trajectory
x, y, theta = initial_state
trajectory = np.zeros((np.arange(0, T, dt).size+1, 3))
trajectory[0] = [x, y, theta]
for t, _ in enumerate(np.arange(0, T, dt)):
    x += v_optimal * np.cos(theta) * dt
    y += v_optimal * np.sin(theta) * dt
    theta += (v_optimal / L) * np.tan(phi_optimal) * dt
    trajectory[t+1] = [x, y, theta]

# Plotting v and phi over time
time_steps = np.arange(0, T, dt)

plt.figure(figsize=(12, 8))

# Plotting the x, y trajectory
plt.subplot(1, 1, 1)
plt.quiver(trajectory[:, 0], trajectory[:, 1], np.cos(trajectory[:, 2]), np.sin(trajectory[:, 2]), scale=100, color='r', label='Orientation')
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Trajectory (x, y)')
plt.scatter(goal_state[0], goal_state[1], color='red', label='Goal', zorder=5)
plt.scatter(initial_state[0], initial_state[1], color='blue', label='Initial', zorder=5)

plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('Optimal Trajectory in the XY Plane')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
