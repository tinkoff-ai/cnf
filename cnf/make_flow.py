from typing import Optional

import torch
import torch.nn as nn

from cnf.flow import (
    NormalizingFlow, InvertibleATanh, Identity, Flow, InvertibleTanh,
    ActNorm, InvertibleLinear, CouplingLayer
)


def make_flow_layers(
        n_layers: int,
        data_dim: int,
        state_emb_dim: int,
        hidden_size: int,
        init_identity: bool = False
):
    layers = []
    for i in range(n_layers):
        layers.append(ActNorm(data_dim, init_identity=init_identity))
        layers.append(InvertibleLinear(data_dim, init_identity=init_identity))
        layers.append(CouplingLayer(i % 2 == 0, data_dim, state_emb_dim, hidden_size, nn.PReLU))
    return layers


def make_flow(
        device: torch.device,
        n_layers: int,
        data_dim: int,
        state_emb_dim: int,
        hidden_size: int,
        add_atanh: bool,
        uniform_latent: bool,
        state_embedding_mean: Optional[torch.Tensor] = None,
        state_embedding_std: Optional[torch.Tensor] = None,
        init_identity: bool = False
) -> NormalizingFlow:
    layers = []
    if add_atanh:
        layers.append(InvertibleATanh())
    else:
        layers.append(Identity())
    layers += make_flow_layers(n_layers, data_dim, state_emb_dim, hidden_size, init_identity=init_identity)
    flow = Flow(layers)

    if uniform_latent:
        # Uniform prior requires tanh to be applied before it
        flow.append_layer(InvertibleTanh())
        latent_min = -1.0 * torch.ones(data_dim, dtype=torch.float32, device=device, requires_grad=False)
        latent_max = torch.ones(data_dim, dtype=torch.float32, device=device, requires_grad=False)
        latent_distribution = torch.distributions.Uniform(latent_min, latent_max)
    else:
        latent_mean = torch.zeros(data_dim, dtype=torch.float32, device=device, requires_grad=False)
        latent_std = torch.ones(data_dim, dtype=torch.float32, device=device, requires_grad=False)
        latent_distribution = torch.distributions.Normal(latent_mean, latent_std)

    normalizing_flow = NormalizingFlow(
        latent_distribution, flow, state_emb_dim, state_embedding_mean, state_embedding_std
    )
    normalizing_flow.to(device)
    return normalizing_flow
