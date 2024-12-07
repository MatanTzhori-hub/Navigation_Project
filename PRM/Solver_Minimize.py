import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class TrajectoryOptimizer:
    def __init__(self, L=2.0, T=1, distance_radius = 10):
        self.L = L  # Wheelbase
        self.T = T  # Time horizon
        self.dt = self.T / 100  # Time step
        self.max_iter = 100    
        self.distance_radius = distance_radius
    
    # weight for edge as its distance
    def edge_road_weight(self, theta1, theta2, phi):
        eps = 1e-5
        phi = phi + eps
        R = np.abs(self.L / np.tan(phi))
        return R * (np.abs(theta1 - theta2))

    def cost_function(self, u, initial_state, goal_state):
        x, y, theta = initial_state
        x_goal, y_goal, theta_goal = goal_state
        v, phi = u[0], u[1]
        cost = 0
        
        if (v <= 0 or v > 30 or np.abs(phi) > np.pi / 3):
            return 50000
        
        for _ in np.arange(0, self.T, self.dt):
            x += v * np.cos(theta) * self.dt
            y += v * np.sin(theta) * self.dt
            theta += (v / self.L) * np.tan(phi) * self.dt
            
        x_diff = (x - x_goal)
        y_diff = (y - y_goal)
        theta_diff = (theta - theta_goal)
        
        cost += x_diff**2 + y_diff**2 + theta_diff**2

        return cost

    def solve(self, initial_state, goal_state, inital_guess=None):
        if inital_guess == None:
            initial_guess = [5, 0]
        result = minimize(self.cost_function, initial_guess, args=(initial_state, goal_state), method='SLSQP', options={'maxiter': self.max_iter})
        v_optimal, phi_optimal = result.x[0], result.x[1]
        return v_optimal, phi_optimal

    def get_trajectory(self, v_optimal, phi_optimal, T_optimal, initial_state, steps=100):
        v_optimal = np.array(v_optimal)
        phi_optimal = np.array(phi_optimal)
        T_optimal = np.array(T_optimal)
        length = v_optimal.size
        initial_state = np.tile(initial_state, (length, 1)).T
        
        x, y, theta = initial_state
        trajectory_x = np.zeros(shape=(length, steps))
        trajectory_y = np.zeros(shape=(length, steps))
        trajectory_theta = np.zeros(shape=(length, steps))
        
        dt = T_optimal / steps
        for i in np.arange(0, steps):
            x += v_optimal * np.cos(theta) * dt
            y += v_optimal * np.sin(theta) * dt
            theta += (v_optimal / self.L) * np.tan(phi_optimal) * dt
            trajectory_x[:, i] = x
            trajectory_y[:, i] = y
            trajectory_theta[:, i] = theta
        return (trajectory_x, trajectory_y, trajectory_theta)

    # def get_trajectory(self, v_optimal, phi_optimal, initial_state):
    #     x, y, theta = initial_state
    #     trajectory_x, trajectory_y, trajectory_theta = [x], [y], [theta]
    #     for _ in np.arange(0, self.T, self.dt):
    #         x += v_optimal * np.cos(theta) * self.dt
    #         y += v_optimal * np.sin(theta) * self.dt
    #         theta += (v_optimal / self.L) * np.tan(phi_optimal) * self.dt
    #         trajectory_x.append(x)
    #         trajectory_y.append(y)
    #         trajectory_theta.append(theta)  
    #     return (trajectory_x, trajectory_y, trajectory_theta)

    def plot_trajectory(self, v_optimal, phi_optimal, T_optimal, initial_state):
        trajectory_x, trajectory_y, trajectory_theta = self.get_trajectory(v_optimal, phi_optimal, T_optimal, initial_state)
        
        plt.quiver(trajectory_x[::10], trajectory_y[::10], np.cos(trajectory_theta[::10]), np.sin(trajectory_theta[::10]), 
                   scale=100, width=0.002, color='r', label='_Hidden label')
        plt.plot(trajectory_x, trajectory_y, color="r")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Trajectory')
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
    