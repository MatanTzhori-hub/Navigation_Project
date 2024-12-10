import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import pandas
import datetime

from scripts import utils

checkpoint_filename = "checkpoints/[3, 128, 128, 128, 3]__2000__100__12_08_14_23_49"
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
L = 2

for i in range(x.shape[0]):
    utils.plot_trajectory(y[i, 0], y[i, 1], y[i, 2], [0,0,0], L, 'b')
    utils.plot_trajectory(y_pred[i, 0], y_pred[i, 1], y_pred[i, 2], [0,0,0], L, 'r')
    plt.quiver(x[i, 0], x[i, 1], 
                    np.cos(x[i, 2]), np.sin(x[i, 2]), 
                    scale=100, width=0.002)

# Add legend with custom labels for blue and red trajectories
plt.title(checkpoint_filename.split('/')[-1])
plt.legend(handles=[Line2D([0], [0], color='b', label='Expected'), Line2D([0], [0], color='r', label='Predicted')])

date = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
# plt.savefig(f"figures/MLP_Sampling/{date}.png")
plt.show()
