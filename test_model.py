import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import pandas
import datetime

from scripts import utils

def get_trajectory(v_optimal, phi_optimal, initial_state, T, L, steps=100):
        x, y, theta = initial_state
        trajectory_x, trajectory_y, trajectory_theta = [x], [y], [theta]
        dt = T / steps
        for _ in np.arange(steps):
            theta += (v_optimal / L) * np.tan(phi_optimal) * dt
            x += v_optimal * np.cos(theta) * dt
            y += v_optimal * np.sin(theta) * dt
            trajectory_x.append(x)
            trajectory_y.append(y)
            trajectory_theta.append(theta)  
        return (trajectory_x, trajectory_y, trajectory_theta)


def plot_trajectory(v_optimal, phi_optimal, initial_state, T, L, color, label=None):
    trajectory_x, trajectory_y, trajectory_theta = get_trajectory(v_optimal, phi_optimal, initial_state, T, L)
    
    plt.quiver(trajectory_x[::10], trajectory_y[::10], 
                np.cos(trajectory_theta[::10]), np.sin(trajectory_theta[::10]), 
                scale=100, width=0.002, color=color, label=label)
    plt.plot(trajectory_x, trajectory_y, color=color)

checkpoint_filename = "checkpoints/model_[3, 64, 64, 64, 3]"
data_path = "dataset/AckermanDataset10K_test.csv"
# data_path = "dataset/overfitting_train.csv"

model = torch.load(checkpoint_filename, weights_only=False)
model.eval()
# print(model)

data = torch.from_numpy(pandas.read_csv(data_path).values)
x = data[:25, 0:3]
y = data[:25, 3:].numpy()

# x = utils.destination(2, data[0, 5], data[0, 3], data[0, 4], torch.tensor([0,0,0]))


y_pred = model(x)
y_pred = y_pred.detach().numpy()

for i in range(x.shape[0]):
    plot_trajectory(y[i, 0], y[i, 1], [0,0,0], y[i, 2], 2, 'b')
    plot_trajectory(y_pred[i, 0], y_pred[i, 1], [0,0,0], y_pred[i, 2], 2, 'r')
    plt.quiver(x[i, 0], x[i, 1], 
                    np.cos(x[i, 2]), np.sin(x[i, 2]), 
                    scale=100, width=0.002)

# Add legend with custom labels for blue and red trajectories
plt.title(checkpoint_filename.split('/')[-1])
plt.legend(handles=[Line2D([0], [0], color='b', label='Expected'), Line2D([0], [0], color='r', label='Predicted')])

date = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
# plt.savefig(f"figures/MLP_Sampling/{date}.png")
plt.show()
