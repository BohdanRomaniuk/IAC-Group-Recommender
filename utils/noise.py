"""Ornstein-Uhlenbeck exploration noise for continuous DDPG actions.

The OU process produces temporally correlated Gaussian noise that is well
suited to scoring-based recommenders where adjacent actions should be
similar.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:  # avoid runtime import cycle
    from config import TrainingConfig


__all__ = ["OrnsteinUhlenbeckNoise"]

logger = logging.getLogger(__name__)


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process for continuous-action exploration.

    The process is defined by the SDE
    ``dx = θ(μ - x) dt + σ dW``
    discretised here with ``dt = 1``.
    """

    def __init__(self, config: "TrainingConfig") -> None:
        """Configure the OU process from a :class:`TrainingConfig`.

        Args:
            config: Source of ``ou_mu``, ``ou_theta``, ``ou_sigma``,
                ``ou_epsilon`` and ``action_embedding_dim``.
        """
        self.action_dim: int = config.action_embedding_dim
        self.mu: float = config.ou_mu
        self.theta: float = config.ou_theta
        self.sigma: float = config.ou_sigma
        self.epsilon: float = config.ou_epsilon
        self._state: torch.Tensor = torch.empty(self.action_dim)
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Reset the process to its mean ``μ``."""
        self._state = torch.full((self.action_dim,), float(self.mu))

    # ------------------------------------------------------------------ #
    def _step(self) -> None:
        """Advance the OU process by one discrete time step."""
        drift = self.theta * (self.mu - self._state)
        diffusion = self.sigma * torch.randn(self.action_dim)
        self._state = self._state + drift + diffusion

    # ------------------------------------------------------------------ #
    def sample(self) -> torch.Tensor:
        """Step the process and return a fresh noise sample.

        Returns:
            A 1-D tensor of shape ``[action_embedding_dim]``.
        """
        self._step()
        return self._state.clone()






