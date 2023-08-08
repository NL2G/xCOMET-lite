from typing import List, Optional

import torch
from torch import nn


class FeedForward(nn.Module):
    """Feed Forward Neural Network.

    Args:
        in_dim (int): Number input features.
        out_dim (int): Number of output features. Default is just a score.
        hidden_sizes (List[int]): List with hidden layer sizes. Defaults to [3072,1024]
        activations (str): Name of the activation function to be used in the hidden
            layers. Defaults to 'Tanh'.
        final_activation (Optional[str]): Final activation if any.
        dropout (float): dropout to be used in the hidden layers.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_sizes: list[int] = [3072, 1024],
        activations: str = "Tanh",
        final_activation: Optional[str] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        modules = []
        modules.append(nn.Linear(in_dim, hidden_sizes[0]))
        modules.append(self.build_activation(activations))
        modules.append(nn.Dropout(dropout))

        for i in range(1, len(hidden_sizes)):
            modules.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            modules.append(self.build_activation(activations))
            modules.append(nn.Dropout(dropout))

        modules.append(nn.Linear(hidden_sizes[-1], int(out_dim)))
        if final_activation is not None:
            modules.append(self.build_activation(final_activation))

        self.ff = nn.Sequential(*modules)

    def build_activation(self, activation: str) -> nn.Module:
        if hasattr(nn, activation.title()):
            return getattr(nn, activation.title())()
        else:
            raise Exception(f"{activation} is not a valid activation function!")

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        return self.ff(in_features)


class LayerwiseAttention(nn.Module):
    def __init__(
        self,
        num_layers: int,
        layer_norm: bool = False,
        layer_weights: List[float] | None = None,
        dropout: float | None = None,
        layer_transformation: str = "softmax",
    ) -> None:
        super(LayerwiseAttention, self).__init__()
        self.num_layers = num_layers
        self.layer_norm = layer_norm
        self.dropout = dropout

        self.transform_fn = torch.softmax
        if layer_transformation == "sparsemax":
            from entmax import sparsemax

            self.transform_fn = sparsemax

        if layer_weights is None:
            layer_weights = [0.0] * num_layers
        elif len(layer_weights) != num_layers:
            raise Exception(
                "Length of layer_weights {} differs \
                from num_layers {}".format(
                    layer_weights, num_layers
                )
            )

        self.scalar_parameters = nn.ParameterList(
            [
                nn.Parameter(
                    torch.FloatTensor([layer_weights[i]]),
                    requires_grad=True,
                )
                for i in range(num_layers)
            ]
        )

        self.gamma = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(
        self,
        tensors: List[torch.Tensor],  # pylint: disable=arguments-differ
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(tensors) != self.num_layers:
            raise Exception(
                "{} tensors were passed, but the module was initialized to \
                mix {} tensors.".format(
                    len(tensors), self.num_layers
                )
            )

        # BUG: Pytorch bug fix when Parameters are not well copied across GPUs
        # https://github.com/pytorch/pytorch/issues/36035
        if len([parameter for parameter in self.scalar_parameters]) != self.num_layers:
            weights = torch.tensor(self.weights, device=tensors[0].device)
            gamma = torch.tensor(self.gamma_value, device=tensors[0].device)
        else:
            weights = torch.cat([parameter for parameter in self.scalar_parameters])
            gamma = self.gamma

        normed_weights = self.transform_fn(weights, dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)

        pieces = []
        for weight, tensor in zip(normed_weights, tensors):
            pieces.append(weight * tensor)
        return gamma * sum(pieces)
