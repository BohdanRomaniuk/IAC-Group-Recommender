"""Recommendation environment driven by an NMF rating predictor.

Provides :class:`RecommendationEnvironment` — a Gymnasium environment whose
hidden reward function is a low-rank reconstruction of the training rating
matrix produced by :class:`sklearn.decomposition.NMF`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import gymnasium as gym
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF

from config import TrainingConfig


__all__ = ["RecommendationEnvironment"]

logger = logging.getLogger(__name__)


_VALID_SPLITS: Tuple[str, ...] = ("train", "val", "test")


class RecommendationEnvironment(gym.Env):
    """Group-recommendation environment with NMF-imputed rewards."""

    metadata = {"render.modes": ["human"]}
    reward_range = (0, 1)

    # ================================================================== #
    def __init__(
        self,
        config: TrainingConfig,
        rating_matrix: csr_matrix,
        dataset_name: str,
    ) -> None:
        """Initialise the environment.

        Args:
            config: Training configuration.
            rating_matrix: Sparse (group, item) → rating matrix.
            dataset_name: Source split — ``'train'``, ``'val'`` or ``'test'``.
        """
        if dataset_name not in _VALID_SPLITS:
            raise ValueError(f"dataset_name must be one of {_VALID_SPLITS}, got {dataset_name!r}")

        self.config = config
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(config.action_size,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(config.state_size,))

        self.rating_matrix = rating_matrix
        coo = rating_matrix.tocoo()
        self.known_rating_indices = set(zip(coo.row.tolist(), coo.col.tolist()))

        self._env_name = f"env_{dataset_name}_{config.env_n_components}.npy"
        self.env_path = str(Path(config.checkpoint_dir) / self._env_name)

        self.predicted_ratings: np.ndarray = np.empty(0)
        self.load_env()

        self.state: List[int] = []
        self.reset()

    # ================================================================== #
    # NMF predictor                                                       #
    # ================================================================== #
    def load_env(self) -> None:
        """Load the cached NMF prediction matrix or fit it on first run."""
        cache_path = Path(self.env_path)
        if cache_path.exists():
            self.predicted_ratings = np.load(cache_path)
            logger.info("Load environment: %s", cache_path)
            return

        logger.info("Train environment (NMF, k=%d)", self.config.env_n_components)
        model = NMF(
            n_components=self.config.env_n_components,
            init="random",
            tol=self.config.env_tol,
            max_iter=self.config.env_max_iter,
            alpha_W=self.config.env_alpha,
            alpha_H=self.config.env_alpha,
            l1_ratio=0.0,
            verbose=True,
            random_state=0,
        )
        w = model.fit_transform(X=self.rating_matrix)
        self.predicted_ratings = w @ model.components_

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, self.predicted_ratings)
        logger.info("Save environment: %s", cache_path)

    # ================================================================== #
    # Gym API                                                             #
    # ================================================================== #
    def reset(self) -> List[int]:
        """Sample a random group with at least ``history_length`` items."""
        rng = np.random.default_rng()
        history_length = self.config.history_length
        total_groups = self.config.total_group_num

        while True:
            group_id = int(rng.integers(low=1, high=total_groups + 1))
            _, nonzero_cols = self.rating_matrix[group_id, :].nonzero()
            if len(nonzero_cols) >= history_length:
                break

        history = rng.choice(nonzero_cols, size=history_length, replace=False).tolist()
        self.state = [group_id] + history
        return self.state

    # ------------------------------------------------------------------ #
    def step(self, action: int) -> Tuple[List[int], int, bool, dict]:
        """Apply ``action`` (an item id) and return ``(state, reward, done, info)``."""
        group_id = self.state[0]
        history = self.state[1:]

        if (group_id, action) in self.known_rating_indices:
            reward = int(self.rating_matrix[group_id, action])
        else:
            p_like = float(self.predicted_ratings[group_id, action])
            p_like = max(0.0, min(1.0, p_like))
            reward = int(np.random.choice(self.config.rewards, p=[1 - p_like, p_like]))

        if reward > 0:
            history = history[1:] + [action]

        self.state = [group_id] + history
        return self.state, reward, False, {}

    # ------------------------------------------------------------------ #
    def render(self, mode: str = "human") -> None:  # pragma: no cover
        """No-op renderer (the environment is not visual)."""
        return None






