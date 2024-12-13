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
        self, in_dim: int, dims: Sequence[int], nonlins: Sequence[Union[str, nn.Module]]
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
        assert len(nonlins) == len(dims)
        self.norm = False
    
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
        
    def normalized(self, to_norm: bool, xmin: Sequence[float], xmax: Sequence[float]):
        self.to_norm = to_norm
        self.xmin = xmin
        self.xmax = xmax
        
    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: An input tensor, of shape (N, D) containing N samples with D features.
        :return: An output tensor of shape (N, D_out) where D_out is the output dim.
        """
        if self.to_norm:
            x = (x - self.xmin) / (self.xmax - self.xmin)
        return self.layers(x)


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2, dropout_rate=0.1):
        super(GatingNetwork, self).__init__()
        
        self.k = top_k
        layers = []
        
        layers.append(nn.Linear(input_dim, 128))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(128, num_experts))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: An input tensor, of shape (N, D) containing N samples with D features.
        :return: An output tensor of shape (N, D_out) where D_out is the number of experts.
        """
        logits = self.layers(x)
        top_k_logits, top_k_indices = logits.topk(self.k, dim=-1)
        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, top_k_indices, top_k_logits)
        gating_output = torch.softmax(sparse_logits, dim=-1)
        
        return gating_output, top_k_indices
        # return torch.softmax(self.layers(x), dim=1)


class MOE(nn.Module):
    """
    Mixture of Experts (MOE) using MLPs.
    """

    def __init__(
        self, in_dim: int, dims: Sequence[int], nonlins: Sequence[Union[str, nn.Module]], num_experts: int=4, top_k=2
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
        super(MOE, self).__init__()
        assert len(nonlins) == len(dims)
        
        self.to_norm = False
        self.in_dim = in_dim
        self.num_experts = num_experts
        self.out_dim = dims[-1]
        
        self.experts = nn.ModuleList([MLP(in_dim, dims, nonlins) for _ in range(num_experts)])
        self.gating = GatingNetwork(in_dim, num_experts, top_k)
        
    def normalized(self, to_norm: bool, xmin: Sequence[float], xmax: Sequence[float]):
        self.to_norm = to_norm
        self.xmin = xmin
        self.xmax = xmax

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: An input tensor, of shape (N, D) containing N samples with D features.
        :return: An output tensor of shape (N, D_out) where D_out is the output dim.
        """
        if self.to_norm:
            x = (x - self.xmin) / (self.xmax - self.xmin)
        
        weights, indices = self.gating(x)
        # final_output = torch.zeros((x.shape[0], self.out_dim)).to(x.device)

        # # Process each expert in parallel
        # for i, expert in enumerate(self.experts):
        #     # Create a mask for the inputs where the current expert is in top-k
        #     #expert_mask = (indices == i).any(dim=-1)
        #     expert_mask = (indices == i).sum(dim=-1)
        #     flat_mask = expert_mask.view(-1).bool()

        #     if flat_mask.sum() > 0:
        #         expert_input = x[flat_mask]
        #         expert_output = expert(expert_input)

        #         # Extract and apply gating scores
        #         gating_scores = gating_output[flat_mask, i].unsqueeze(1)
        #         weighted_output = expert_output * gating_scores

        #         # Update final output additively by indexing and adding
        #         final_output[flat_mask] += weighted_output.squeeze(1)

        # return final_output, indices
        
        # weights = self.gating(x)
        outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        weights = weights.unsqueeze(1).expand_as(outputs)

        return torch.sum(outputs * weights, dim=2)