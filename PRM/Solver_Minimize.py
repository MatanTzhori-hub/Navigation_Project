import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class TrajectoryOptimizer:
    def __init__(self, L=2.0, T=1, dt=0.01):
        # Define parameters
        self.L = L  # Wheelbase
        self.T = T  # Time horizon
        self.dt = dt  # Time step
        self.max_iter = 100

    
    def destination(self, v, phi, initial_state):
        eps = 1e-5
        phi = phi + eps
        
        x_s, y_s, theta_s = initial_state
        
        theta_f = theta_s + (v / self.L) * np.tan(phi) * self.T
        
        x_f = self.L/np.tan(phi)*(np.sin(theta_f)-np.sin(theta_s)) + x_s
        y_f = self.L/np.tan(phi)*(-np.cos(theta_f)+np.cos(theta_s)) + y_s
        
        return x_f,y_f,theta_f
    
    def slice_range(self, theta1, theta2, phi):
        eps = 1e-5
        phi = phi + eps
        
        R = np.abs(self.L / np.tan(phi))
        return R * (np.abs(theta1 - theta2))

    def cost_function(self, u, initial_state, goal_state):
        x, y, theta = initial_state
        theta_init = theta
        x_goal, y_goal, theta_goal = goal_state
        v, phi = u[0], u[1]
        cost = 0
        
        if (v <= 0 or np.abs(phi) > np.pi/3):
            return 50000
        
        # for _ in np.arange(0, self.T, self.dt):
        #     x += v * np.cos(theta) * self.dt
        #     y += v * np.sin(theta) * self.dt
        #     theta += (v / self.L) * np.tan(phi) * self.dt
        # cost += (x - x_goal)**2 + (y - y_goal)**2 + (theta - theta_goal)**2
        
        x_dest, y_dest, theta_dest = self.destination(v, phi, initial_state)
        cost += (x_dest - x_goal)**2 + (y_dest - y_goal)**2 + (theta_dest - theta_goal)**2
        # cost += self.slice_range(theta_init, theta_dest, phi)**2
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

    def plot_trajectory(self, v_optimal, phi_optimal, initial_state, goal_state):
        x, y, theta = initial_state
        trajectory_x, trajectory_y, trajectory_theta = [x], [y], [theta]
        for _ in np.arange(0, self.T, self.dt):
            x += v_optimal * np.cos(theta) * self.dt
            y += v_optimal * np.sin(theta) * self.dt
            theta += (v_optimal / self.L) * np.tan(phi_optimal) * self.dt
            trajectory_x.append(x)
            trajectory_y.append(y)
            trajectory_theta.append(theta)
        
        # plt.quiver(trajectory_x[::5], trajectory_y[::5], np.cos(trajectory_theta[::5]), np.sin(trajectory_theta[::5]), scale=100, color='r', label='_Hidden label')
        plt.plot(trajectory_x, trajectory_y, color="r")
        # plt.scatter(initial_state[0], initial_state[1], color='blue', label='_Hidden label')
        # plt.scatter(goal_state[0], goal_state[1], color='red', label='_Hidden label')
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
    