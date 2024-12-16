import os
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt
import yaml
# from torch.utils.tensorboard import SummaryWriter

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
    title="",
    remove_outliner = False
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
            ax.set_xlabel("Epoch #")
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
        
        if remove_outliner:
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            ax.set_ylim(lower_bound, upper_bound)

        if legend or train_test_overlay:
            ax.legend()
        ax.grid(True)

    min_index = fit_res.test_loss.index(min(fit_res.test_loss))

    plt.suptitle(f"""{title}
                Train: Avg. Loss {fit_res.train_loss[min_index]:.3f}, Avg. XY Distance {fit_res.train_xy_dist[min_index]:.3f}, Avg. Theta Error {fit_res.train_theta_diff[min_index]:.3f}
                Test:  Avg. Loss {fit_res.test_loss[min_index]:.3f}, Avg. XY Distance {fit_res.test_xy_dist[min_index]:.3f}, Avg. Theta Error {fit_res.test_theta_diff[min_index]:.3f}""")
    return fig, axes


def main():
    logdir = "logs"
    # writer = SummaryWriter(logdir)
    writer = None
    
    model_types = ["MOE"]#, "MLP"]
    experts_amounts = [3, 5, 7]
    k_values = [2, 2, 2]
    normalize = [True, False]
    batch_sizes = [5000]
    datasets = [50]
    #hidden_dims = [32, 64, 128]
    hidden_dims = [128, 256]
    learning_rates = [0.005]
    
    p = itertools.product(model_types, normalize, hidden_dims, batch_sizes, datasets, learning_rates)
    for combo in p:
        model_type, norm, dim, batch_size, ds_size, leaning_rate = combo
        for i in range(len(k_values)):
            if i > 0 and model_type != 'MOE':
                continue
            for j in range(1, 2):
                
                path_to_ds = f"dataset/Below_15/AckermanDataset{ds_size}K"
                dl_params = {'batch_size': batch_size, 'shuffle': True}
                train_ds, train_dl, test_ds, test_dl, x_min, x_max = dataset_load.create_dataloaders(path_to_ds, **dl_params)
                
                ## MLP params:
                in_dim = 3
                out_dim = 3
                
                dims = [dim] * j + [out_dim]
                depth = len(dims)
                nonlinear = ["relu"] * depth
                
                ## Optimizer params:
                reg = 0
                
                if model_type == "MLP":
                    model = mlp.MLP(in_dim=in_dim, dims=dims, nonlins=nonlinear)
                elif model_type == "MOE":
                    model = mlp.MOE(in_dim=in_dim, dims=dims, nonlins=nonlinear, num_experts=experts_amounts[i], top_k=k_values[i])
                else:
                    raise NotImplementedError("No such model")
                model.normalized(normalize, x_min, x_max)
                model = model.double()

                loss_fn = torch.nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=leaning_rate, weight_decay=reg, amsgrad=False)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=35, verbose=True)
                # scheduler= None
                
                epochs = 300
                date = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                model_name = f'{model_type}_{[in_dim] + dims}__{date}'
                model_name = f'{model_name}_norm' if norm else model_name
                save_dir = f'runs/{model_name}'
                os.makedirs(save_dir, exist_ok=True)
                plot_samples_dir = f'{save_dir}/Samples_Figs'
                checkpoint = f'{save_dir}/model_{model_name}'
                early_stopping = 100
                print_every = 10
                
                trainer = training.Trainer(model, loss_fn, optimizer, scheduler, writer)
                fit_res = trainer.fit(train_dl, test_dl, epochs, checkpoints=checkpoint,
                                    early_stopping=early_stopping, print_every=print_every, plot_samples=plot_samples_dir)
                
                fig, ax = plot_fit(fit_res, title=f"Model {model_type} {[in_dim] + dims}")
                plt.savefig(f"{save_dir}/{model_name}.png")
                plt.close()
                fig, ax = plot_fit(fit_res, title=f"Model {model_type} {[in_dim] + dims}", remove_outliner=True)
                plt.savefig(f"{save_dir}/{model_name}_OL.png")
                plt.close()
                
                yaml_dump = {
                    'date':date, 
                    'model': {'type': model_type, 'hidden_dims': f'{[in_dim] + dims}', 'depth': depth, 'activation_func': 'ReLU'},
                    'is_normalized': norm, 'batch_size': batch_size, 'dataset': path_to_ds, 'dataset_size': ds_size,
                    'Training Params': {
                        'learning_rate': leaning_rate, 'early_stop': early_stopping, 'max_epochs': epochs, 
                        'loss_func': type(loss_fn).__name__, 'optimizer': type(optimizer).__name__, 'scheduler': type(scheduler).__name__,
                        'sched_factor': scheduler.factor, 'sched_patience': scheduler.patience, 
                    }
                }
                if model_type == 'MOE':
                    yaml_dump['model'].update({'num_experts': experts_amounts[i], 'top_k': k_values[i]})
                    
                with open(f'{save_dir}/configuration.yaml', 'w') as file:
                    yaml.dump(yaml_dump, file, default_flow_style=False, sort_keys=False)
        
if __name__ == "__main__":
    main()
    