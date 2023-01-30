from typing import Tuple

import torch
import torch.nn as nn


class ActNorm(nn.Module):
    def __init__(
            self,
            input_size: int,
            init_identity: bool = False
    ):
        super().__init__()
        self._input_size = input_size
        # it is better to have this variable registered because it will be saved/loaded correctly.
        initialized = torch.tensor(init_identity, dtype=torch.bool)
        self.register_buffer('_initialized', initialized)
        self._log_scale = nn.Parameter(torch.zeros(1, input_size), requires_grad=True)
        self._translate = nn.Parameter(torch.zeros(1, input_size), requires_grad=True)

    def _initialize(self, data: torch.Tensor, forward_mode: bool):
        data = data.view(-1, self._input_size)
        log_std = torch.log(data.std(dim=0, keepdim=True) + 1e-6)
        mean = data.mean(dim=0, keepdim=True)

        if forward_mode:
            self._log_scale.data = -log_std.data
            self._translate.data = (-mean * torch.exp(-log_std)).data
        else:
            self._log_scale.data = log_std.data
            self._translate.data = mean.data
        self._initialized = torch.tensor(True, dtype=torch.bool)

    def f(self, x: torch.Tensor, *_) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._initialized and self.training:
            self._initialize(x, forward_mode=True)
        z = torch.exp(self._log_scale) * x + self._translate
        log_prob = (self._log_scale * torch.ones_like(x)).sum(-1)
        return z, log_prob

    def g(self, z: torch.Tensor, *_) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._initialized and self.training:
            self._initialize(z, forward_mode=False)
        x = (z - self._translate) * torch.exp(-self._log_scale)
        log_prob = (self._log_scale * torch.ones_like(z)).sum(-1)
        return x, log_prob
