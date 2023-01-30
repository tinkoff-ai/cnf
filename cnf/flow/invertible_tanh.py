from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional


eps = 1e-5


class InvertibleATanh(nn.Module):
    """Ideologically it must be the first layer of the flow,
    because it will transform data via atanh during forward (i.e. data -> noise)
    and apply tanh during inverse (i.e. noise -> data)
    """
    @staticmethod
    def f(x: torch.Tensor, *_) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.clamp(x, -1 + eps, 1 - eps)
        z = 0.5 * torch.log((1 + x) / (1 - x))
        log_prob = -torch.log(1 - x ** 2).sum(-1)
        return z, log_prob

    @staticmethod
    def g(z: torch.Tensor, *_) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.clamp(torch.tanh(z), -1 + eps, 1 - eps)
        log_prob = -torch.log(1.0 - x ** 2.0).sum(-1)
        return x, log_prob


class InvertibleTanh(nn.Module):
    """Useful for controlling latent distribution shape, with this layer it may become Uniform"""
    @staticmethod
    def f(x: torch.Tensor, *_) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.clamp(torch.tanh(x), -1 + eps, 1 - eps)
        log_prob = torch.log(1.0 - z ** 2.0).sum(-1)
        return z, log_prob

    @staticmethod
    def g(z: torch.Tensor, *_) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.clamp(z, -1 + eps, 1 - eps)
        x = 0.5 * torch.log((1 + z) / (1 - z))
        log_prob = torch.log(1 - z ** 2).sum(-1)
        return x, log_prob
