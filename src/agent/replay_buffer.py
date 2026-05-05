"""Prioritised Experience Replay (proportional variant).

Implements :class:`PrioritizedExperienceReplay` (Schaul et al., ICLR 2016).
Transitions with larger TD-error are sampled more often; importance-sampling
(IS) weights — annealed via the ``β`` schedule — correct the resulting bias
during the gradient update.
"""
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import torch


__all__ = ["PrioritizedExperienceReplay"]


Experience = Tuple[Any, Any, Any, Any]
SampleTuple = Tuple[List[Experience], np.ndarray, torch.Tensor]


class PrioritizedExperienceReplay:
    """Proportional Prioritised Experience Replay buffer."""

    def __init__(
        self,
        buffer_size: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
    ) -> None:
        """Allocate the storage and priority arrays.

        Args:
            buffer_size: Maximum number of stored transitions.
            alpha: Priority exponent — ``0`` is uniform, ``1`` is full PER.
            beta_start: Initial IS-weight exponent (annealed to 1).
            beta_frames: Number of :meth:`add` calls over which ``β``
                reaches 1.0.
        """
        self.buffer_size: int = int(buffer_size)
        self.alpha: float = float(alpha)
        self.beta: float = float(beta_start)
        self._beta_increment: float = (1.0 - beta_start) / max(beta_frames, 1)

        self._storage: List[Experience] = []
        self._priorities: np.ndarray = np.zeros(self.buffer_size, dtype=np.float32)
        self._cursor: int = 0  # circular write pointer

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._storage)

    # ------------------------------------------------------------------ #
    def add(self, experience: Experience) -> None:
        """Insert a transition with the current maximum priority.

        Args:
            experience: A ``(state, action, reward, next_state)`` tuple.
        """
        if self._storage:
            max_priority = float(self._priorities[: len(self._storage)].max())
        else:
            max_priority = 1.0

        if len(self._storage) < self.buffer_size:
            self._storage.append(experience)
        else:
            self._storage[self._cursor] = experience

        self._priorities[self._cursor] = max_priority
        self._cursor = (self._cursor + 1) % self.buffer_size



    # ------------------------------------------------------------------ #
    def sample(self, batch_size: int) -> SampleTuple:
        """Sample a batch proportionally to stored priorities.

        Args:
            batch_size: Number of transitions to draw.

        Returns:
            ``(batch, indices, importance_weights)`` where ``batch`` is a
            list of experiences, ``indices`` is a numpy array used by
            :meth:`update_priorities`, and ``importance_weights`` is a 1-D
            tensor of normalised IS weights.
        """
        n = len(self._storage)
        priorities = self._priorities[:n]

        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)

        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # Anneal β
        self.beta = min(1.0, self.beta + self._beta_increment)

        batch = [self._storage[i] for i in indices]
        is_weights = torch.from_numpy(weights.astype(np.float32))
        return batch, indices, is_weights

    # ------------------------------------------------------------------ #
    def update_priorities(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray,
        epsilon: float = 1e-6,
    ) -> None:
        """Refresh priorities for a previously sampled batch.

        Args:
            indices: Indices returned by :meth:`sample`.
            td_errors: Absolute TD errors for those transitions.
            epsilon: Small floor to avoid zero priorities.
        """
        new_priorities = np.abs(td_errors).astype(np.float32) + float(epsilon)
        self._priorities[indices] = new_priorities






