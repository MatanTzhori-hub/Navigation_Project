import torch
import numpy as np
import os.path
import matplotlib.pyplot as plt

import utils

def show_hist(velocity, steering, time):
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))  # 3 rows, 1 column

        # Plot the first histogram
        axs[0].hist(velocity, bins=30, alpha=0.7, color='blue')
        axs[0].set_title('Velocity Distribution')
        axs[0].grid()

        # Plot the second histogram
        axs[1].hist(steering, bins=30, alpha=0.7, color='green')
        axs[1].set_title('Steering angle Distribution')
        axs[1].set_xlabel('Value')
        axs[1].set_ylabel('Density')
        axs[1].grid()

        # Plot the third histogram
        axs[2].hist(time, bins=30, alpha=0.7, color='red')
        axs[2].set_title('Time Distribution')
        axs[2].set_xlabel('Value')
        axs[2].set_ylabel('Density')
        axs[2].grid()

        # Adjust layout
        plt.tight_layout()

        # Show the plots
        plt.show()

gen_n_samples = 100000  # num of samples to generate
n_samples = 100       # num of samples to save

plot_hist = False
ratio = 0.2
titles = 'deltaX,deltaY,deltaTheta,Velocity,SteeringAngle,Time'
filespath = 'dataset'
# train_file = 'AckermanDataset10K_train.csv'
# test_file = 'AckermanDataset10K_test.csv'
train_file = 'overfitting_train.csv'
test_file = 'overfitting_test.csv'

max_delta = torch.pi/4
min_v = 5
max_v = 10
min_T = 0.1
max_T = 1.5
L = 2
start = [0,0,0]

train_path = os.path.join(filespath, train_file)
test_path = os.path.join(filespath, test_file)

start = torch.Tensor(start)

v = min_v + (max_v-min_v)*torch.rand(gen_n_samples)
delta = -max_delta + (max_delta-(-max_delta))*torch.rand(gen_n_samples)
T = min_T + (max_T-min_T)*torch.rand(gen_n_samples)

goal = utils.destination(L, T, v, delta, start)

dataset = torch.stack((*goal, v, delta, T), axis=1)
filter_idxs = (-torch.pi < dataset[:, 2]) & (dataset[:, 2] < torch.pi)
dataset = dataset[filter_idxs]
np.savetxt("dataset/BigDS_Filttered.csv", dataset, delimiter=',', header=titles, comments='', fmt='%.6f')

if plot_hist:
        show_hist(dataset[:,3].numpy(), dataset[:,4].numpy(), dataset[:,5].numpy())

# save only 10k
dataset = dataset[0:n_samples, :]
    
test_size = (int)((torch.floor(torch.tensor(n_samples*ratio))).item())
train_size = n_samples - test_size

train_ds = dataset[:train_size, :].numpy()
test_ds = dataset[train_size:, :].numpy()

np.savetxt(train_path, train_ds, delimiter=',', header=titles, comments='', fmt='%.6f')
np.savetxt(test_path, test_ds, delimiter=',', header=titles, comments='', fmt='%.6f')
    