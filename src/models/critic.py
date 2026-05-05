"""Q-value network used by the DDPG critic.

The :class:`Critic` estimates the action-value :math:`Q(s, a)` given an
embedded state and an embedded action. The output is unbounded (no
activation on the final layer) — required for valid Bellman updates.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


__all__ = ["Critic"]


_DROPOUT_RATE: float = 0.1


def _block(in_features: int, out_features: int, *, dropout: bool = True) -> nn.Sequential:
    """Build a ``Linear → LayerNorm → ReLU [→ Dropout]`` block."""
    layers: list[nn.Module] = [
        nn.Linear(in_features, out_features),
        nn.LayerNorm(out_features),
        nn.ReLU(inplace=True),
    ]
    if dropout:
        layers.append(nn.Dropout(p=_DROPOUT_RATE))
    return nn.Sequential(*layers)


class Critic(nn.Module):
    """Critic network producing a scalar Q-value per (state, action) pair.

    Architecture (default sizes):

    ``(704 + 64) → 256 → 256 → 128 → 1``
    """

    def __init__(
        self,
        embedded_state_size: int,
        embedded_action_size: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        """Build the critic MLP.

        Args:
            embedded_state_size: Dimensionality of the encoded state.
            embedded_action_size: Dimensionality of the encoded action.
            hidden_sizes: Hidden-layer sizes. At least two values required.
        """
        super().__init__()

        if len(hidden_sizes) < 2:
            raise ValueError("Critic requires at least two hidden sizes")

        h0, h1 = hidden_sizes[0], hidden_sizes[1]
        input_dim = embedded_state_size + embedded_action_size

        self.net = nn.Sequential(
            _block(input_dim, h0),
            _block(h0, h0),                # extra hidden layer
            _block(h0, h1, dropout=False),
            nn.Linear(h1, 1),              # unbounded Q-value
        )

    # ------------------------------------------------------------------ #
    def forward(
        self,
        embedded_state: torch.Tensor,
        embedded_action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute :math:`Q(s, a)`.

        Args:
            embedded_state: Tensor of shape ``[B, embedded_state_size]``.
            embedded_action: Tensor of shape ``[B, embedded_action_size]``.

        Returns:
            Q-value tensor of shape ``[B, 1]``.
        """
        joint = torch.cat([embedded_state, embedded_action], dim=-1)
        return self.net(joint)

