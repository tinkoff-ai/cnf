from typing import Optional

import torch
import torch.nn as nn


class NormLayer(nn.Module):
    """Special layer which performs state/observation normalization."""
    def __init__(
            self,
            state_embedding_dim: int,
            state_embedding_mean: Optional[torch.Tensor] = None,
            state_embedding_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if state_embedding_mean is None:
            state_embedding_mean = torch.zeros(state_embedding_dim, dtype=torch.float32)
        self._state_embedding_mean = nn.Parameter(state_embedding_mean, requires_grad=False)

        if state_embedding_std is None:
            state_embedding_std = torch.ones(state_embedding_dim, dtype=torch.float32)
        self._state_embedding_std = nn.Parameter(state_embedding_std, requires_grad=False)

    def forward(self, state_embedding):
        if state_embedding is not None:
            if self._state_embedding_mean is not None:
                state_embedding = state_embedding - self._state_embedding_mean
            if self._state_embedding_std is not None:
                state_embedding = state_embedding / self._state_embedding_std
        return state_embedding
