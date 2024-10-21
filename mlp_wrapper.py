import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt

from MLP import mlp
from MLP import training
from MLP.training import FitResult
from MLP import dataset_load


def plot_fit(
    fit_res: FitResult,
    fig=None,
    log_loss=False,
    legend=None,
    train_test_overlay: bool = False,
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

    return fig, axes


def main():
    batch_size = 64
    path_to_ds = "dataset/"
    dl_params = {'batch_size': batch_size, 'shuffle': True}
    train_ds, train_dl, test_ds, test_dl = dataset_load.create_dataloaders(path_to_ds, **dl_params)
    
    ## MLP params:
    in_dim = 3
    dims = [64, 64, 3]
    depth = len(dims)
    nonlinear = ["relu"] * depth
    
    ## Optimizer params:
    leaning_rate = 0.05
    reg = 0
    
    model = mlp.MLP(in_dim=in_dim, dims=dims, nonlins=nonlinear)
    model = model.double()
    ## List of regression loss functions
    loss_functions = {
        'Mean Squared Error Loss': torch.nn.MSELoss(),
        'Mean Absolute Error Loss': torch.nn.L1Loss(),
        'Huber Loss': torch.nn.SmoothL1Loss(),
        'Poisson Loss': torch.nn.PoissonNLLLoss(),
    }

    loss_fn = loss_functions['Mean Squared Error Loss']
    optimizer = torch.optim.Adam(model.parameters(), lr=leaning_rate, weight_decay=reg, amsgrad=False)
    
    epochs = 100
    checkpoint = 'checkpoints/model_checkpoint'
    early_stopping = 15
    print_every = 1
    
    trainer = training.LayerTrainer(model, loss_fn, optimizer)
    fit_res = trainer.fit(train_dl, test_dl, epochs, checkpoints=checkpoint,
                          early_stopping=early_stopping, print_every=print_every)
    
    fig, ax = plot_fit(fit_res)
    plt.show()
    
if __name__ == "__main__":
    main()