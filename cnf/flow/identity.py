from typing import Tuple

import torch
import torch.nn as nn


class Identity(nn.Module):
    @staticmethod
    def f(x: torch.Tensor, *_) -> Tuple[torch.Tensor, float]:
        return x, 0.0

    @staticmethod
    def g(z: torch.Tensor, *_) -> Tuple[torch.Tensor, float]:
        return z, 0.0
