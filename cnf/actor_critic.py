from typing import Optional

import torch
import torch.nn as nn

from cnf.mlp import MLP
from cnf.flow import NormLayer


class TanhActor(nn.Module):
    def __init__(
            self,
            state_size: int,
            action_size: int,
            hidden_size: int,
            state_mean: Optional[torch.Tensor] = None,
            state_std: Optional[torch.Tensor] = None,
            n_layers: int = 3,
    ):
        super().__init__()
        self._norm_layer = NormLayer(state_size, state_mean, state_std)
        self._mlp = MLP(
            state_size, 2 * action_size, hidden_size, nn.ReLU,
            init_zeros=False, n_layers=n_layers
        )

    def _get_base_policy(self, state):
        state = self._norm_layer(state)
        mean_log_std = self._mlp(state)
        mean, log_std = torch.chunk(mean_log_std, 2, dim=-1)
        log_std = log_std.clamp(-20.0, 2.0)
        policy = torch.distributions.Normal(mean, log_std.exp())
        return policy

    @staticmethod
    def _log_prob(policy, base_action, tanh_action):
        base_log_prob = policy.log_prob(base_action).sum(-1, keepdim=True)
        tanh_log_prob = - torch.log(1.0 - tanh_action ** 2 + 1e-6).sum(-1, keepdim=True)
        log_prob = base_log_prob + tanh_log_prob
        return log_prob

    def act(self, state):
        base_policy = self._get_base_policy(state)
        if self.training:
            base_action = base_policy.rsample()
        else:
            base_action = base_policy.mean

        tanh_action = torch.tanh(base_action)
        log_prob = self._log_prob(base_policy, base_action, tanh_action)

        return tanh_action, log_prob

    def log_prob(self, state, tanh_action):
        tanh_action.clamp_(-0.999, +0.999)
        base_policy = self._get_base_policy(state)
        base_action = torch.atan(tanh_action)
        log_prob = self._log_prob(base_policy, base_action, tanh_action)
        return log_prob


class Critic(nn.Module):
    def __init__(
            self,
            state_size: int,
            action_size: int,
            hidden_size: int,
            state_mean: Optional[torch.Tensor] = None,
            state_std: Optional[torch.Tensor] = None,
            n_layers: int = 3,
            add_layer_norm: bool = False,
    ):
        super().__init__()
        self._norm_layer = NormLayer(state_size, state_mean, state_std)
        self._mlp = MLP(
            state_size + action_size, 1, hidden_size, nn.ReLU,
            init_zeros=False, n_layers=n_layers, add_layer_norm=add_layer_norm
        )

    def forward(self, state, action):
        state = self._norm_layer(state)
        q_value = self._mlp(state, action)
        return q_value
