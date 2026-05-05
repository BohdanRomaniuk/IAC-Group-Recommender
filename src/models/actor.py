"""Deterministic policy network used by the DDPG actor.

The :class:`Actor` consumes an embedded state vector and emits a single
continuous *action weight* in ``[-1, 1]`` whose dimensionality matches the
item-embedding space, allowing the agent to score candidate items by inner
product.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


__all__ = ["Actor"]


_DROPOUT_RATE: float = 0.1


def _mlp_block(in_features: int, out_features: int, *, dropout: bool = True) -> nn.Sequential:
    """Build a ``Linear → LayerNorm → ReLU [→ Dropout]`` block."""
    layers: list[nn.Module] = [
        nn.Linear(in_features, out_features),
        nn.LayerNorm(out_features),
        nn.ReLU(inplace=True),
    ]
    if dropout:
        layers.append(nn.Dropout(p=_DROPOUT_RATE))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Actor network with LayerNorm + Dropout regularisation.

    Architecture (default sizes):

    ``704 → 256 → 256 → 128 → 64 (Tanh)``
    """

    def __init__(
        self,
        embedded_state_size: int,
        action_weight_size: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        """Build the actor MLP.

        Args:
            embedded_state_size: Dimensionality of the encoded state.
            action_weight_size: Dimensionality of the output action weight
                (matches the item-embedding dimensionality).
            hidden_sizes: Hidden-layer sizes. At least two values are
                expected; the first is reused once for the extra hidden
                layer that improves expressiveness.
        """
        super().__init__()

        if len(hidden_sizes) < 2:
            raise ValueError("Actor requires at least two hidden sizes")

        h0, h1 = hidden_sizes[0], hidden_sizes[1]
        self.net = nn.Sequential(
            _mlp_block(embedded_state_size, h0),
            _mlp_block(h0, h0),               # extra hidden layer
            _mlp_block(h0, h1),
            nn.Linear(h1, action_weight_size),
            nn.Tanh(),                        # bound to [-1, 1]
        )

    # ------------------------------------------------------------------ #
    def forward(self, embedded_state: torch.Tensor) -> torch.Tensor:
        """Map an embedded state to an action-weight vector.

        Args:
            embedded_state: Tensor of shape ``[..., embedded_state_size]``.

        Returns:
            Action-weight tensor of shape ``[..., action_weight_size]``.
        """
        return self.net(embedded_state)

