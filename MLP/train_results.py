from typing import List, NamedTuple


class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch: the loss
    and number of correct classifications.
    """

    loss: float
    theta_diff_mean: float
    distance_mean: float


class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch: the loss per batch
    and accuracy on the dataset (train or test).
    """

    losses: List[float]
    theta_diff_mean: List[float]
    distance_mean: List[float]


class FitResult(NamedTuple):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch and the accuracies are per epoch.
    """

    num_epochs: int
    train_loss: List[float]
    train_xy_dist: List[float]
    train_theta_diff: List[float]
    test_loss: List[float]
    test_xy_dist: List[float]
    test_theta_diff: List[float]
