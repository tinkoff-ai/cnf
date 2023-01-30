from typing import Tuple

import torch
import torch.nn as nn


class InvertibleLinear(nn.Module):
    # A simple invertible linear layer with no LU decomposition,
    # analogue of invertible 1x1 convolution from the Glow paper.
    # Since all matrices are small, LU decomposition
    # is (probably) not very important.
    def __init__(
            self,
            input_size: int,
            init_identity: bool = False
    ):
        super().__init__()
        if init_identity:
            w = torch.diag(torch.ones(input_size, dtype=torch.float32))
        else:
            # q is an orthogonal matrix which implies that it is a rotation matrix.
            w, _ = torch.linalg.qr(torch.randn(input_size, input_size))
        # shouldn't I force W to be such that det(W) != 0?
        # I think yes, but in practice it works fine w/o this constraint.
        self._weight = nn.Parameter(w)

    def f(self, x: torch.Tensor, *_) -> Tuple[torch.Tensor, torch.Tensor]:
        z = nn.functional.linear(x, self._weight)
        w = self._weight.to(torch.device('cpu'))
        _, log_prob = torch.linalg.slogdet(w)
        log_prob = log_prob.to(x.device)
        return z, log_prob

    def g(self, z: torch.Tensor, *_) -> Tuple[torch.Tensor, torch.Tensor]:
        w = self._weight.to(torch.device('cpu'))
        w_inv = torch.linalg.inv(w).to(z.device)
        x = nn.functional.linear(z, w_inv)
        _, log_prob = torch.linalg.slogdet(w)
        log_prob = log_prob.to(z.device)
        return x, log_prob
