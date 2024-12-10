import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from MLP import mlp
from MLP import training
from MLP.training import FitResult
from MLP import dataset_load

import datetime


def plot_fit(
    fit_res: FitResult,
    fig=None,
    log_loss=False,
    legend=None,
    train_test_overlay: bool = False,
    title=""
):
    """
    Plots a FitResult object.
    Creates four plots: train loss, test loss, train acc, test acc.
    :param fit_res: The fit result to plot.
    :param fig: A figure previously returned from this function. If not None,
        plots will the added to this figure.
    :param log_loss: Whether to plot the losses in log scale.
    :param legend: What to call this FitResult in the legend.
    :param train_test_overlay: Whether to overlay train/test plots on the same axis.
    :return: The figure.
    """
    if fig is None:
        nrows = 1 if train_test_overlay else 2
        ncols = 3
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(8 * ncols, 5 * nrows),
            sharex="col",
            sharey=False,
            squeeze=False,
        )
        axes = axes.reshape(-1)
    else:
        axes = fig.axes

    for ax in axes:
        for line in ax.lines:
            if line.get_label() == legend:
                line.remove()

    p = itertools.product(enumerate(["train", "test"]), enumerate(["loss", "xy_dist", "theta_diff"]))
    for (i, traintest), (j, loss_avg) in p:

        ax = axes[j if train_test_overlay else i * 3 + j]

        attr = f"{traintest}_{loss_avg}"
        data = getattr(fit_res, attr)
        label = traintest if train_test_overlay else legend
        h = ax.plot(np.arange(1, len(data) + 1), data, label=label)
        ax.set_title(attr)

        if loss_avg == "loss":
            ax.set_xlabel("Iteration #")
            ax.set_ylabel("Loss")
            if log_loss:
                ax.set_yscale("log")
                ax.set_ylabel("Loss (log)")
        elif loss_avg == "xy_dist":
            ax.set_xlabel("Epoch #")
            ax.set_ylabel("XY Mean Distance Error")
        else:
            ax.set_xlabel("Epoch #")
            ax.set_ylabel("Mean Theta Error")


        if legend or train_test_overlay:
            ax.legend()
        ax.grid(True)

    plt.suptitle(f"""{title}\n 
                Train: Avg. Loss {fit_res.train_loss[-1]:.3f}, Avg. XY Distance {fit_res.train_xy_dist[-1]:.3f}, Avg. Theta Error {fit_res.train_theta_diff[-1]:.3f}\n
                Test:  Avg. Loss {fit_res.test_loss[-1]:.3f}, Avg. XY Distance {fit_res.test_xy_dist[-1]:.3f}, Avg. Theta Error {fit_res.test_theta_diff[-1]:.3f}\n""")
    return fig, axes


def main():
    logdir = "logs"
    writer = SummaryWriter(logdir)
    
    normalize = False
    batch_sizes = [250, 500, 1000, 2000]
    datasets = [10, 30, 50, 70, 100]

    #hidden_dims = [32, 64, 128]
    hidden_dims = [32, 64, 128]
    for i, dim in enumerate(hidden_dims):
        for j in range(1, 4):
            for batch_size in batch_sizes:
                for ds_size in datasets:
                    path_to_ds = f"dataset/Below_15/AckermanDataset{ds_size}K"
                    dl_params = {'batch_size': batch_size, 'shuffle': True}
                    train_ds, train_dl, test_ds, test_dl = dataset_load.create_dataloaders(path_to_ds, **dl_params)
                    
                    ## MLP params:
                    in_dim = 3
                    out_dim = 3
                    
                    dims = [dim] * j + [out_dim]
                    depth = len(dims)
                    nonlinear = ["relu"] * depth
                    
                    ## Optimizer params:
                    leaning_rate = 0.005
                    reg = 0
                    
                    model = mlp.MLP(in_dim=in_dim, dims=dims, nonlins=nonlinear, norm=normalize)
                    model = model.double()
                    print(model)

                    loss_fn = torch.nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=leaning_rate, weight_decay=reg, amsgrad=False)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=35, verbose=True)
                    # scheduler= None
                    
                    epochs = 201
                    date = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                    checkpoint = f'checkpoints/model_{[in_dim] + dims}'
                    checkpoint = checkpoint + '_norm' if normalize else checkpoint
                    checkpoint = checkpoint + f'__{date}'
                    plot_samples_dir = f'figures/MLP_Training/{date}'
                    early_stopping = 100
                    print_every = 10
                    
                    plt.ioff()
                    trainer = training.Trainer(model, loss_fn, optimizer, scheduler, writer)
                    fit_res = trainer.fit(train_dl, test_dl, epochs, checkpoints=checkpoint,
                                        early_stopping=early_stopping, print_every=print_every, plot_samples=plot_samples_dir)
                    plt.ion()
                    
                    fig, ax = plot_fit(fit_res, title=f"Model layers (left -> right): {[in_dim] + dims}")
                    
                    plt.savefig(f"figures/MLP_Training/{date}/{[in_dim] + dims}__{batch_size}__{ds_size}__.png")
                    # plt.show()
        
if __name__ == "__main__":
    main()
    