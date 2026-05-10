"""Training configuration for the IAC group recommender.

Defines :class:`TrainingConfig`, a dataclass that centralises every
hyper-parameter, file-system path and runtime device used across the project.
A YAML loader is provided for reproducible experiments.

Backward-compatibility aliases (legacy names such as ``Config``,
``saves_folder_path``, ``num_episodes``…) are exposed at the bottom of
this module so existing imports keep working without modification.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import torch

try:  # PyYAML is optional
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


__all__ = ["TrainingConfig"]

logger = logging.getLogger(__name__)


_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_DATASET_DIR = _PROJECT_ROOT / "data" / "MovieLens-1m"
_DEFAULT_CHECKPOINT_DIR = Path("generators/data") / "saves"
_DEFAULT_OUTPUT_DIR = Path("generators/data") / "results"

_TUPLE_FIELDS = frozenset({"actor_hidden_sizes", "critic_hidden_sizes"})


def _resolve_device() -> torch.device:
    """Return the best torch device available at runtime."""
    if torch.cuda.is_available():
        logger.info("Using GPU (CUDA acceleration enabled)")
        return torch.device("cuda:0")
    logger.info("Using CPU")
    return torch.device("cpu")


@dataclass
class TrainingConfig:
    """Project-wide configuration container.

    All hyper-parameters live here so that scripts can be driven by a
    single typed object. Derived fields (``state_size``,
    ``state_embedding_dim``, ``action_embedding_dim``) are computed in
    :meth:`__post_init__`.
    """

    # --- Paths ----------------------------------------------------------- #
    dataset_dir: str = str(_DEFAULT_DATASET_DIR)
    checkpoint_dir: str = str(_DEFAULT_CHECKPOINT_DIR)
    output_dir: str = str(_DEFAULT_OUTPUT_DIR)

    # --- Recommendation system ------------------------------------------ #
    history_length: int = 10
    eval_top_k_values: List[int] = field(default_factory=lambda: [5, 10, 15, 20])
    rewards: List[int] = field(default_factory=lambda: [0, 1])

    # --- Reinforcement learning ----------------------------------------- #
    embedding_size: int = 64
    action_size: int = 1

    # --- Dataset cardinalities (filled at runtime) ---------------------- #
    item_num: Optional[int] = None
    user_num: Optional[int] = None
    group_num: Optional[int] = None
    total_group_num: Optional[int] = None

    # --- Environment (NMF) --------------------------------------------- #
    env_tol: float = 1e-4
    env_max_iter: int = 1000
    env_alpha: float = 1e-3

    # --- Networks ------------------------------------------------------- #
    actor_hidden_sizes: Tuple[int, ...] = (256, 128)
    critic_hidden_sizes: Tuple[int, ...] = (256, 128)

    # --- DDPG ---------------------------------------------------------- #
    tau: float = 5e-3
    gamma: float = 0.95

    # --- Optimisation -------------------------------------------------- #
    batch_size: int = 128
    buffer_size: int = 100_000
    max_episodes: int = 1000
    steps_per_episode: int = 100
    gradient_steps_per_env_step: int = 2
    warmup_steps: int = 500
    embedding_weight_decay: float = 1e-6
    actor_weight_decay: float = 1e-6
    critic_weight_decay: float = 1e-6
    embedding_learning_rate: float = 5e-4
    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    eval_interval_episodes: int = 10

    # --- OU noise ------------------------------------------------------ #
    ou_mu: float = 0.0
    ou_theta: float = 0.15
    ou_sigma: float = 0.3
    ou_epsilon: float = 1.0

    # --- Entropy regularisation --------------------------------------- #
    entropy_coef: float = 0.05

    # --- Reward shaping ------------------------------------------------ #
    reward_diversity_alpha: float = 0.5
    reward_coverage_beta: float = 0.1

    # --- Prioritised replay ------------------------------------------- #
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100_000

    # --- Derived (set in __post_init__) ------------------------------- #
    state_size: int = field(init=False)
    state_embedding_dim: int = field(init=False)
    action_embedding_dim: int = field(init=False)
    env_n_components: int = field(init=False)
    device: torch.device = field(init=False)

    # ------------------------------------------------------------------ #
    # Mapping of legacy attribute names → new names (used by from_yaml)   #
    # ------------------------------------------------------------------ #
    _LEGACY_FIELD_MAP: ClassVar[Dict[str, str]] = {
        "data_folder_path": "dataset_dir",
        "saves_folder_path": "checkpoint_dir",
        "results_folder_path": "output_dir",
        "top_K_list": "eval_top_k_values",
        "num_episodes": "max_episodes",
        "num_steps": "steps_per_episode",
        "num_updates_per_step": "gradient_steps_per_env_step",
        "eval_per_iter": "eval_interval_episodes",
    }

    # ================================================================== #
    def __post_init__(self) -> None:
        self._recompute_derived()
        self.device = _resolve_device()

    def _recompute_derived(self) -> None:
        """(Re)compute fields derived from primary hyper-parameters."""
        self.state_size = self.history_length + 1
        self.state_embedding_dim = self.state_size * self.embedding_size
        self.action_embedding_dim = self.action_size * self.embedding_size
        self.env_n_components = self.embedding_size

    # ------------------------------------------------------------------ #
    # Path helpers                                                        #
    # ------------------------------------------------------------------ #
    @property
    def item_path(self) -> str:
        """Path to ``movies.dat``."""
        return str(Path(self.dataset_dir) / "movies.dat")

    @property
    def user_path(self) -> str:
        """Path to ``users.dat``."""
        return str(Path(self.dataset_dir) / "users.dat")

    @property
    def group_path(self) -> str:
        """Path to ``groupMember.dat``."""
        return str(Path(self.dataset_dir) / "groupMember.dat")

    # ------------------------------------------------------------------ #
    # YAML loading                                                        #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: Optional[str]) -> "TrainingConfig":
        """Load a config from YAML, falling back to defaults for missing keys.

        Args:
            path: Path to a YAML file. ``None`` returns the defaults.

        Returns:
            A populated :class:`TrainingConfig` instance.
        """
        if not path:
            return cls()
        if yaml is None:
            raise ImportError(
                "PyYAML is required for TrainingConfig.from_yaml(); "
                "install it via `pip install pyyaml`."
            )

        with open(path, "r", encoding="utf-8") as fh:
            payload: Dict[str, Any] = yaml.safe_load(fh) or {}

        valid_field_names = {f.name for f in fields(cls) if f.init}
        kwargs: Dict[str, Any] = {}

        for raw_key, raw_value in payload.items():
            key = cls._LEGACY_FIELD_MAP.get(raw_key, raw_key)
            if key not in valid_field_names:
                logger.warning("Ignoring unknown config key: %s", raw_key)
                continue
            value = (
                tuple(raw_value)
                if key in _TUPLE_FIELDS and isinstance(raw_value, list)
                else raw_value
            )
            kwargs[key] = value

        return cls(**kwargs)




