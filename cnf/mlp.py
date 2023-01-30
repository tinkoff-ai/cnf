from typing import Callable

import torch
import torch.nn as nn


class MLP(nn.Module):
    # 3-layer MLP with ReLU activation
    def __init__(
            self,
            input_size: int,
            output_size: int,
            hidden_size: int,
            activation: Callable[[], nn.Module],
            init_zeros: bool = True,
            n_layers: int = 3,
            add_layer_norm: bool = False
    ):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size)]
        if add_layer_norm:
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(activation())
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if add_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(activation())
        layers.append(nn.Linear(hidden_size, output_size))

        if init_zeros:
            layers[-1].bias.data.zero_()
            layers[-1].weight.data.zero_()
        self._mlp = nn.Sequential(*layers)

    def forward(self, *x: torch.Tensor):
        mlp_input = torch.cat([xx for xx in x if xx is not None], dim=-1)
        return self._mlp(mlp_input)
