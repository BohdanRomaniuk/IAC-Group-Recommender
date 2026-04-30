"""Uniform replay buffer used as a simple alternative to PER.

Provides :class:`UniformReplayBuffer` backed by ``collections.deque`` —
O(1) append, O(n) sampling without replacement.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Any, Deque, List, Tuple


__all__ = ["UniformReplayBuffer"]


# Type alias: (state, action, reward, next_state)
Experience = Tuple[Any, Any, Any, Any]


class UniformReplayBuffer:
    """Bounded FIFO buffer that samples transitions uniformly at random."""

    def __init__(self, buffer_size: int) -> None:
        """Allocate a buffer with the given maximum capacity.

        Args:
            buffer_size: Maximum number of stored transitions.
        """
        self.buffer_size: int = int(buffer_size)
        self._storage: Deque[Experience] = deque(maxlen=self.buffer_size)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._storage)

    # ------------------------------------------------------------------ #
    def add(self, experience: Experience) -> None:
        """Append a single ``(state, action, reward, next_state)`` tuple."""
        self._storage.append(experience)



    # ------------------------------------------------------------------ #
    def sample_batch(self, batch_size: int) -> List[Experience]:
        """Draw ``batch_size`` transitions without replacement.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            A list of experiences.
        """
        return random.sample(self._storage, batch_size)






