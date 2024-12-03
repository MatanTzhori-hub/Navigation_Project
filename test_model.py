import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas

from scripts import utils

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


def plot_trajectory(v_optimal, phi_optimal, initial_state, T, dt, L, color, label):
    trajectory_x, trajectory_y, trajectory_theta = get_trajectory(v_optimal, phi_optimal, initial_state, T, dt, L)
    
    plt.quiver(trajectory_x[::10], trajectory_y[::10], 
                np.cos(trajectory_theta[::10]), np.sin(trajectory_theta[::10]), 
                scale=100, width=0.002, color=color, label=label)
    plt.plot(trajectory_x, trajectory_y, color=color)

checkpoint_filename = "checkpoints/model_checkpoint_[3, 256, 256, 256, 256, 3]"
data_path = "dataset/AckermanDataset10K_test.csv"

model = torch.load(checkpoint_filename)
# print(model)

data = torch.from_numpy(pandas.read_csv(data_path).values)
x = data[:25, 0:3]
y = data[:25, 3:].numpy()

# x = utils.destination(2, data[0, 5], data[0, 3], data[0, 4], torch.tensor([0,0,0]))


y_pred = model(x)
y_pred = y_pred.detach().numpy()

for i in range(x.shape[0]):
    plot_trajectory(y[i, 0], y[i, 1], [0,0,0], y[i, 2], y[i, 2] / 100, 2, 'b', "expected")
    plot_trajectory(y_pred[i, 0], y_pred[i, 1], [0,0,0], y_pred[i, 2], y[i, 2] / 100, 2, 'r', "predicted")
    plt.quiver(x[i, 0], x[i, 1], 
                    np.cos(x[i, 2]), np.sin(x[i, 2]), 
                    scale=100, width=0.002, label='goal')
plt.legend()
plt.show()