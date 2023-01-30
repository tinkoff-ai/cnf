from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from cnf.flow.norm_state_layer import NormLayer


class Flow(nn.Module):
    # stack of normalizing flow layers
    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self._layers = nn.ModuleList(layers)

    def flow_forward(
            self,
            x: torch.Tensor,
            state_embedding: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x -> z; data -> latent
        z = x
        log_prob = torch.zeros(*x.size()[:-1], dtype=torch.float32, device=x.device)
        for layer in self._layers:
            z, layer_log_prob = layer.f(z, state_embedding)
            log_prob += layer_log_prob
        return z, log_prob

    def flow_inverse(
            self,
            z: torch.Tensor,
            state_embedding: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = z
        log_prob = torch.zeros(*z.size()[:-1], dtype=torch.float32, device=z.device)
        for layer in reversed(self._layers):
            x, layer_log_prob = layer.g(x, state_embedding)
            log_prob += layer_log_prob
        return x, log_prob

    def append_layer(self, layer: nn.Module):
        self._layers.append(layer)

    def append_layers(self, layers: List[nn.Module]):
        self._layers.extend(layers)


class NormalizingFlow(nn.Module):
    def __init__(
            self,
            latent_distribution: torch.distributions.Distribution,
            flow: Flow,
            state_embedding_dim: int,
            state_embedding_mean: Optional[torch.Tensor] = None,
            state_embedding_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self._latent_distribution = latent_distribution
        self._flow = flow
        self._state_embedding_norm_layer = NormLayer(
            state_embedding_dim, state_embedding_mean, state_embedding_std
        )

    def flow_forward(
            self,
            x: torch.Tensor,
            state_embedding: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x -> z; data -> latent
        state_embedding = self._state_embedding_norm_layer(state_embedding)
        z, log_prob = self._flow.flow_forward(x, state_embedding)
        log_prob += self._latent_distribution.log_prob(z).sum(-1)
        return z, log_prob

    def flow_inverse(
            self,
            z: torch.Tensor,
            state_embedding: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # z -> x; latent -> data.
        state_embedding = self._state_embedding_norm_layer(state_embedding)
        log_prob = self._latent_distribution.log_prob(z).sum(-1)

        x, flow_log_prob = self._flow.flow_inverse(z, state_embedding)
        log_prob += flow_log_prob

        return x, log_prob

    def sample(
            self,
            sample_shape: Union[int, torch.Size],
            state_embedding: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # samples a new data point from the flow distribution.
        z = self._latent_distribution.sample(sample_shape)
        x, log_prob = self.flow_inverse(z, state_embedding)
        return x, log_prob

    def log_prob(
            self,
            x: torch.Tensor,
            state_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # computes log-probability for a data point.
        _, log_prob = self.flow_forward(x, state_embedding)
        return log_prob

    def append_layer(self, layer: nn.Module):
        self._flow.append_layer(layer)

    def append_layers(self, layers: List[nn.Module]):
        self._flow.append_layers(layers)
