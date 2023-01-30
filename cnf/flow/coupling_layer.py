from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from cnf.mlp import MLP


class CouplingLayer(nn.Module):
    def __init__(
            self,
            parity: bool,
            input_size: int,
            state_emb_size: int,
            hidden_size: int,
            activation: Callable[[], nn.Module]
    ):
        super().__init__()
        self._rescale = nn.Parameter(torch.ones(input_size // 2))
        self._parity = parity
        self._scale_translate_net = MLP(
            input_size // 2 + state_emb_size, input_size, hidden_size, activation
        )

    def f(
            self,
            x: torch.Tensor,
            state_embedding: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = torch.chunk(x, 2, dim=-1)
        if self._parity:
            x0, x1 = x1, x0

        log_scale_translate = self._scale_translate_net(x0, state_embedding)
        log_scale, translate = torch.chunk(log_scale_translate, 2, dim=-1)
        log_scale = self._rescale * torch.tanh(log_scale)
        z0 = x0
        z1 = torch.exp(log_scale) * x1 + translate

        if self._parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=-1)
        log_det = log_scale.sum(-1)

        return z, log_det

    def g(
            self,
            z: torch.Tensor,
            state_embedding: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z0, z1 = torch.chunk(z, 2, dim=-1)
        if self._parity:
            z0, z1 = z1, z0

        log_scale_translate = self._scale_translate_net(z0, state_embedding)
        log_scale, translate = torch.chunk(log_scale_translate, 2, dim=-1)
        log_scale = self._rescale * torch.tanh(log_scale)
        x0 = z0
        x1 = (z1 - translate) * torch.exp(-log_scale)

        if self._parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=-1)
        log_det = torch.sum(log_scale, dim=-1)

        return x, log_det
