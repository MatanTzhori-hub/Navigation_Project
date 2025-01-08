import torch
import numpy as np
import os.path
import matplotlib.pyplot as plt

import utils

def show_hist(dataset):
        fig, axs = plt.subplots(2, 3, figsize=(10, 12))
        axs = axs.reshape(-1)
        
        # Plot Velocity histogram
        axs[0].hist(dataset[:, 3], bins=30, alpha=0.7, color='blue')
        axs[0].set_title('Velocity Distribution')
        axs[0].set_xlabel('m/s')
        axs[0].grid()

        # Plot Steering histogram
        axs[1].hist(dataset[:, 4], bins=30, alpha=0.7, color='green')
        axs[1].set_title('Steering angle Distribution')
        axs[1].set_xlabel('Rad')
        axs[1].grid()

        # Plot Time histogram
        axs[2].hist(dataset[:, 5], bins=30, alpha=0.7, color='red')
        axs[2].set_title('Time Distribution')
        axs[2].set_xlabel('Seconds')
        axs[2].grid()
        
        # Plot X histogram
        axs[3].hist(dataset[:, 0], bins=30, alpha=0.7, color=plt.cm.tab10(0))
        axs[3].set_title('X Distribution')
        axs[3].set_xlabel('Meter')
        axs[3].grid()
        
        # Plot Y histogram
        axs[4].hist(dataset[:, 1], bins=30, alpha=0.7, color=plt.cm.tab10(0))
        axs[4].set_title('Y Distribution')
        axs[4].set_xlabel('Meter')
        axs[4].grid()
        
        # Plot Theta histogram
        axs[5].hist(dataset[:, 2], bins=30, alpha=0.7, color=plt.cm.tab10(0))
        axs[5].set_title('Theta Distribution')
        axs[5].set_xlabel('Meter')
        axs[5].grid()

        # Adjust layout
        plt.tight_layout()

        # Show the plots
        plt.show()
        
        
def show_dist_hist(dataset):
        fig, axs = plt.subplots(1, 1, figsize=(10, 12))
        # axs = axs.reshape(-1)
        
        # Plot Distance histogram
        distances = np.sqrt(dataset[:, 0]**2 + dataset[:, 1]**2)
        
        axs.hist(distances, bins=30, alpha=0.7, color=plt.cm.tab10(0))
        axs.set_title('Distance Distribution')
        axs.set_xlabel('Meter')
        axs.grid()
        # Adjust layout
        plt.tight_layout()

        # Show the plots
        plt.show()
     
def plot_reach_scatter(dataset, start):
        plt.quiver(dataset[:, 0], dataset[:, 1], 
                np.cos(dataset[:, 2]), np.sin(dataset[:, 2]), 
                scale=200, width=0.002, color="purple", label='_Hidden label')
        plt.quiver(start[0], start[1], 
                np.cos(start[2]), np.sin(start[2]), 
                scale=100, width=0.002, color="blue", label='_Hidden label')

        # Show the plot
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Reach')
        plt.grid(True)
        plt.show()   
         
        
gen_n_samples = 100000  # num of samples to generate
n_samples = 100000       # num of samples to save

plot_hist = True
plot_scatter = True
ratio = 0.2
titles = 'deltaX,deltaY,deltaTheta,Velocity,SteeringAngle,Time'
filespath = 'dataset'
train_file = 'AckermanDataset10K_train.csv'
test_file = 'AckermanDataset10K_test.csv'
# train_file = 'overfitting_train.csv'
# test_file = 'overfitting_test.csv'

max_delta = np.pi/4
min_v = 5
max_v = 10
min_T = 0.1
max_T = 3.5
L = 2
start = [0,0,0]

train_path = os.path.join(filespath, train_file)
test_path = os.path.join(filespath, test_file)

start = np.array(start)

v = min_v + (max_v-min_v)*np.random.rand(gen_n_samples)
delta = -max_delta + (max_delta-(-max_delta))*np.random.rand(gen_n_samples)
T = min_T + (max_T-min_T)*np.random.rand(gen_n_samples)

goal = utils.destination(L, T, v, delta, start)

dataset = np.stack((*goal, v, delta, T), axis=1)
# Filter samples that circle more than half a circle
filter_idxs = (-np.pi < dataset[:, 2]) & (dataset[:, 2] < np.pi)
dataset = dataset[filter_idxs]
# Filter samples with distance smaller than 50
filter_idxs = (15 > np.sqrt(dataset[:, 0]**2 + dataset[:, 1]**2))
dataset = dataset[filter_idxs]

# save only n_samples
dataset = dataset[0:n_samples, :]

if plot_hist:
        show_hist(dataset)
        show_dist_hist(dataset)
if plot_scatter:
        plot_reach_scatter(dataset, start)
    
test_size = (int)((np.floor(n_samples*ratio)))
train_size = n_samples - test_size

train_ds = dataset[:train_size, :]
test_ds = dataset[train_size:, :]

np.savetxt(train_path, train_ds, delimiter=',', header=titles, comments='', fmt='%.6f')
np.savetxt(test_path, test_ds, delimiter=',', header=titles, comments='', fmt='%.6f')
    