import torch
import numpy as np
import matplotlib.pyplot as plt


def limit_by_distance(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2) < 0.05

def get_trajectory(v_optimal, phi_optimal, T_optimal, initial_state, end_state, L, steps=100):
    dtype = np.float64
    v_optimal = np.array(v_optimal, dtype=dtype)
    phi_optimal = np.array(phi_optimal, dtype=dtype)
    T_optimal = np.array(T_optimal, dtype=dtype)
    length = v_optimal.size
    initial_state = np.tile(np.array(initial_state, dtype=dtype), (length, 1)).T
    end_state = np.atleast_2d(np.array(end_state, dtype=dtype))
    
    x, y, theta = initial_state
    trajectory_x = np.zeros(shape=(length, steps))
    trajectory_y = np.zeros(shape=(length, steps))
    trajectory_theta = np.zeros(shape=(length, steps))
    
    dt = T_optimal / steps
    for i in np.arange(0, steps):
        theta += (v_optimal / L) * np.tan(phi_optimal) * dt
        x += v_optimal * np.cos(theta) * dt
        y += v_optimal * np.sin(theta) * dt
        trajectory_x[:, i] = x
        trajectory_y[:, i] = y
        trajectory_theta[:, i] = theta
        
        flag = limit_by_distance(x, y, end_state[:, 0], end_state[:, 1])
        if any(flag):
            if np.isscalar(dt):
                T_optimal = (i+1) * dt
            else:
                T_optimal[flag] = (i+1) * dt[flag]
        
    return (trajectory_x, trajectory_y, trajectory_theta), T_optimal


def plot_trajectory(v_optimal, phi_optimal, T_optimal, initial_state, L, color=None, label=None):
    trajectory, _ = get_trajectory(v_optimal, phi_optimal, T_optimal, initial_state, [0, 0, 0], L)
    trajectory_x, trajectory_y, trajectory_theta = trajectory
    
    for traj_x, traj_y, traj_theta in zip(trajectory_x, trajectory_y, trajectory_theta):
        plt.quiver(traj_x[::10], traj_y[::10], np.cos(traj_theta[::10]), np.sin(traj_theta[::10]), 
                    scale=100, width=0.002, color=color, label=label, zorder=10)
        plt.plot(traj_x, traj_y, color=color, zorder=10)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Trajectory')
        plt.grid(True)


#optimal destination after T seconds in ackerman model
def destination(L, T, v, phi, initial_state):
    eps = 1e-5
    phi = phi + eps 
    x_s, y_s, theta_s = initial_state
    theta_f = theta_s + (v / L) * np.tan(phi) * T
    x_f = L/np.tan(phi)*(np.sin(theta_f)-np.sin(theta_s)) + x_s
    y_f = L/np.tan(phi)*(-np.cos(theta_f)+np.cos(theta_s)) + y_s
    return x_f,y_f,theta_f