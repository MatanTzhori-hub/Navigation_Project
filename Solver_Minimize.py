import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class TrajectoryOptimizer:
    def __init__(self, L=2.0, T=1, dt=0.01):
        # Define parameters
        self.L = L  # Wheelbase
        self.T = T  # Time horizon
        self.dt = dt  # Time step
        self.max_iter = 1000

    def cost_function(self, u, initial_state, goal_state):
        x, y, theta = initial_state
        x_goal, y_goal, theta_goal = goal_state
        v, phi = u[0], u[1]
        cost = 0
        for _ in np.arange(0, self.T, self.dt):
            x += v * np.cos(theta) * self.dt
            y += v * np.sin(theta) * self.dt
            theta += (v / self.L) * np.tan(phi) * self.dt
        cost += (x - x_goal)**2 + (y - y_goal)**2 + (theta - theta_goal)**2
        return cost

    def solve(self, initial_state=None, goal_state=None, inital_guess=None):
        # Initial guess for control inputs (flattened array of shape (2,))
        assert initial_state is not None
        assert goal_state is not None
        
        if inital_guess == None:
            inital_guess = [5, 0]

        # Optimize
        result = minimize(self.cost_function, inital_guess, args=(initial_state, goal_state), method='SLSQP', options={'maxiter': self.max_iter})

        # Extract v and phi
        v_optimal, phi_optimal = result.x[0], result.x[1]
        return v_optimal, phi_optimal

    def plot_trajectory(self, v_optimal, phi_optimal, initial_state, goal_state, initial_guess):
        x, y, theta = initial_state
        trajectory_x, trajectory_y, trajectory_theta = [x], [y], [theta]
        for _ in np.arange(0, self.T, self.dt):
            x += v_optimal * np.cos(theta) * self.dt
            y += v_optimal * np.sin(theta) * self.dt
            theta += (v_optimal / self.L) * np.tan(phi_optimal) * self.dt
            trajectory_x.append(x)
            trajectory_y.append(y)
            trajectory_theta.append(theta)
        
        plt.quiver(trajectory_x, trajectory_y, np.cos(trajectory_theta), np.sin(trajectory_theta), scale=100, color='r', label='_Hidden label')
        plt.plot(trajectory_x, trajectory_y, label=f'init_gues={initial_guess}')
        plt.scatter(initial_state[0], initial_state[1], color='blue', label='_Hidden label')
        plt.scatter(goal_state[0], goal_state[1], color='red', label='_Hidden label')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Trajectory')
        plt.legend()
        plt.grid(True)

# Usage example
if __name__ == "__main__":
    optimizer = TrajectoryOptimizer()
    initial_state = [-5, 5, 0]
    goal_state = [5, 10, np.pi/4]
    initial_guess = [5, 0]
    
    v_optimal, phi_optimal = optimizer.solve(initial_state, goal_state, initial_guess)
    print(f"Optimal v: {v_optimal}, Optimal phi: {phi_optimal}")

    # Plot the optimal trajectory
    optimizer.plot_trajectory(v_optimal, phi_optimal, initial_state, goal_state)
    