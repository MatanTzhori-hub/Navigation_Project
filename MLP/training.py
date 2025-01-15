import os
import abc
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import torch.nn as nn
import torch.utils
import tqdm.auto
from typing import Any, Callable, Optional
from torch.utils.data import DataLoader

from .train_results import FitResult, BatchResult, EpochResult
from scripts import utils

class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn,
        optimizer,
        scheduler = None,
        writer = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer
        self.device = device

        if self.device:
            model.to(self.device)

    def destination_error(self, x, y_pred):
        """
        Checks that res_pred is inside the threshold of eps in comparison to res.
        :param res_pred: Models prediction
        :param res: Samples labels
        :param eps: the threshold
        :return: Binary vector, 1 if sample is in the threshold, otherwise 0.
        """
        x = x.detach().numpy()
        y_pred = y_pred.detach().numpy()
        L = 2
        start = np.array([0, 0, 0])
        end = np.stack(utils.destination(L, y_pred[:, 2], y_pred[:, 0], y_pred[:, 1], start), axis=1)

        theta_diff_mean = np.mean(np.abs(x[:, 2] - end[:, 2]))
        xy_mean = np.mean(np.sqrt(np.sum(((x[:, 0:2] - end[:, 0:2])**2), axis=1)))

        return xy_mean, theta_diff_mean
    
    def plot_samples(self, x, y, dir, epoch, info):
        self.model.eval()
        y_pred = self.model(x)
        y_pred = y_pred.detach().numpy()
        self.model.train()
        plt.figure(figsize=(16,9))
        utils.plot_trajectory(y[:, 0], y[:, 1], y[:, 2], [0,0,0], 2, 'b')
        utils.plot_trajectory(y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], [0,0,0], 2, 'r')
        plt.title(f"""Test samples, epoch: {epoch}
                      Train: Avg. Loss {info[0]:.3f}, Avg. XY {info[1]:.3f}, Avg. Theta {info[2]:.3f}
                      Test: Avg. Loss {info[3]:.3f}, Avg. XY {info[4]:.3f}, Avg. Theta {info[5]:.3f}""")
        plt.legend(handles=[Line2D([0], [0], color='b', label='Expected'), Line2D([0], [0], color='r', label='Predicted')])
        plt.savefig(f"{dir}/epoch_{epoch}")
        plt.close()
        

    def fit(
        self,
        dl_train: DataLoader,
        dl_test: DataLoader,
        num_epochs: int,
        checkpoints: str = None,
        early_stopping: int = None,
        print_every: int = 1,
        plot_samples: str = None,
        **kw,
    ) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param plot_samples: Save sampling plots every few epochs in the given directory.
        :return: A FitResult object containing train and test losses per epoch.
        """
        sample_every = 25
        if plot_samples is not None:
            plt.ioff()
            os.mkdir(plot_samples)
        
        actual_num_epochs = 0
        epochs_without_improvement = 0

        train_loss, train_xy_dist, train_theta_diff, test_loss, test_xy_dist, test_theta_diff = [], [], [], [], [], []
        best_loss = None

        for epoch in range(num_epochs):
            verbose = False  # pass this to train/test_epoch.
            if print_every > 0 and (
                epoch % print_every == 0 or epoch == num_epochs - 1
            ):
                verbose = True
            self._print(f"--- EPOCH {epoch+1}/{num_epochs} ---", verbose)

            train_result = self.train_epoch(dl_train, **kw)
            test_result = self.test_epoch(dl_test, **kw)
            
            train_loss_epoch = sum(train_result.losses).item() / len(dl_train.dataset)
            train_xy_dist_epoch = sum(train_result.distance_mean).item() / len(dl_train.dataset)
            train_theta_diff_epoch = sum(train_result.theta_diff_mean).item() / len(dl_train.dataset)
            train_loss.append(train_loss_epoch)
            train_xy_dist.append(train_xy_dist_epoch)
            train_theta_diff.append(train_theta_diff_epoch)
            
            test_loss_epoch = sum(test_result.losses).item() / len(dl_test.dataset)
            test_xy_dist_epoch = sum(test_result.distance_mean).item() / len(dl_test.dataset)
            test_theta_diff_epoch = sum(test_result.theta_diff_mean).item() / len(dl_test.dataset)
            test_loss.append(test_loss_epoch)
            test_xy_dist.append(test_xy_dist_epoch)
            test_theta_diff.append(test_theta_diff_epoch)

            if self.writer:
                self.writer.add_scalars('Loss', {'train': train_loss_epoch, 'test': test_loss_epoch}, epoch)
                self.writer.add_scalars('XY_Distance', {'train': train_xy_dist_epoch, 'test': test_xy_dist_epoch}, epoch)
                self.writer.add_scalars('Theta_Difference', {'train': train_theta_diff_epoch, 'test': test_theta_diff_epoch}, epoch)
            
            if best_loss is None or test_loss_epoch < best_loss:
                best_loss = test_loss_epoch
                epochs_without_improvement = 0
                
                if checkpoints is not None:
                    self.save_checkpoint(checkpoints)
            else:
                epochs_without_improvement += 1
                if(early_stopping is not None and epochs_without_improvement >= early_stopping):
                    break

            if (plot_samples is not None) and (epoch % sample_every == 0):
                info = (train_loss[-1], train_xy_dist[-1], train_theta_diff[-1],
                        test_loss[-1], test_xy_dist[-1], test_theta_diff[-1])
                x, y = dl_test.dataset[:25]
                self.plot_samples(x, y, plot_samples, epoch, info)
            
            if self.scheduler is not None:
                self.scheduler.step(train_loss_epoch)


        if plot_samples is not None:
            plt.ion()
        return FitResult(actual_num_epochs, train_loss=train_loss, train_xy_dist=train_xy_dist, train_theta_diff=train_theta_diff, test_loss=test_loss, test_xy_dist=test_xy_dist, test_theta_diff=test_theta_diff)

    def save_checkpoint(self, checkpoint_filename: str):
        """
        Saves the model in it's current state to a file with the given name (treated
        as a relative path).
        :param checkpoint_filename: File name or relative path to save to.
        """
        torch.save(self.model, checkpoint_filename)
        print(f"\n*** Saved checkpoint {checkpoint_filename}")

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    def train_batch(self, batch) -> BatchResult:
        X, y = batch

        self.optimizer.zero_grad()
        out = self.model.forward(X)
        batch_loss = self.loss_fn(out, y)
        batch_loss.backward()
        self.optimizer.step()

        xy_dist, theta_error = self.destination_error(X, out)

        return BatchResult(batch_loss, distance_mean=xy_dist ,theta_diff_mean=theta_error)

    def test_batch(self, batch) -> BatchResult:
        X, y = batch
        
        out = self.model.forward(X)
        batch_loss = self.loss_fn(out, y)

        xy_dist, theta_error = self.destination_error(X, out)

        return BatchResult(batch_loss, distance_mean=xy_dist ,theta_diff_mean=theta_error)

    @staticmethod
    def _print(message, verbose=True):
        """Simple wrapper around print to make it conditional"""
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
        dl: DataLoader,
        forward_fn: Callable[[Any], BatchResult],
        verbose=True,
        max_batches=None,
    ) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        theta_diff_mean = []
        distance_mean = []

        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_fn = tqdm.auto.tqdm
            pbar_file = sys.stdout
        else:
            pbar_fn = tqdm.tqdm
            pbar_file = open(os.devnull, "w")

        pbar_name = forward_fn.__name__
        with pbar_fn(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f"{pbar_name} ({batch_res.loss:.3f})")
                pbar.update()

                d = dl.batch_size
                losses.append(batch_res.loss * dl.batch_size)
                theta_diff_mean.append(batch_res.theta_diff_mean * dl.batch_size)
                distance_mean.append(batch_res.distance_mean * dl.batch_size)

            l = len(dl.dataset)
            avg_loss = sum(losses) / len(dl.dataset)
            avg_theta_diff = sum(theta_diff_mean) / len(dl.dataset)
            avg_distance = sum(distance_mean) / len(dl.dataset)
            
            pbar.set_description(
                f"{pbar_name} "
                f"(Avg. Loss {avg_loss:.3f}, "
                f"Avg. Theta diff {avg_theta_diff:.2f}, "
                f"Avg. Distance {avg_distance:.2f})"
            )

        if not verbose:
            pbar_file.close()

        return EpochResult(losses=losses, theta_diff_mean=theta_diff_mean, distance_mean=distance_mean)

