import torch
from torch import Tensor, nn
from typing import Union, Sequence
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax,
    "lrelu": nn.LeakyReLU,
    "none": nn.Identity,
    None: nn.Identity,
}


# Default keyword arguments to pass to activation class constructors, e.g.
# activation_cls(**ACTIVATION_DEFAULT_KWARGS[name])
ACTIVATION_DEFAULT_KWARGS = defaultdict(
    dict,
    {
        ###
        "softmax": dict(dim=1),
        "logsoftmax": dict(dim=1),
    },
)


class MLP(nn.Module):
    """
    A general-purpose MLP.
    """

    def __init__(
        self, in_dim: int, dims: Sequence[int], nonlins: Sequence[Union[str, nn.Module]], norm=False
    ):
        """
        :param in_dim: Input dimension.
        :param dims: Hidden dimensions, including output dimension.
        :param nonlins: Non-linearities to apply after each one of the hidden
            dimensions.
            Can be either a sequence of strings which are keys in the ACTIVATIONS
            dict, or instances of nn.Module (e.g. an instance of nn.ReLU()).
            Length should match 'dims'.
        """
        self.norm = norm
        
        assert len(nonlins) == len(dims)
        self.in_dim = in_dim
        self.out_dim = dims[-1]
        super().__init__()
        layers = []
        dims = [in_dim] + dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
            if i < len(dims) - 2:
                # layers.append(nn.BatchNorm1d(dims[i + 1]))
                if nonlins[i] in ACTIVATIONS:
                    layers.append(ACTIVATIONS[nonlins[i]](
                        **ACTIVATION_DEFAULT_KWARGS[nonlins[i]]))
                else:
                    layers.append(nonlins[i])
                # layers.append(nn.Dropout(p=0.03))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: An input tensor, of shape (N, D) containing N samples with D features.
        :return: An output tensor of shape (N, D_out) where D_out is the output dim.
        """
        if self.norm:
            scaler = MinMaxScaler().fit(x)
            x = torch.Tensor(scaler.transform(x)).to(torch.float64)
        return self.layers(x)
