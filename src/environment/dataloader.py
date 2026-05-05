"""Data loading utilities for the MovieLens group-recommendation datasets.

Provides :class:`MovieLensDatasetLoader` which parses the ``.dat`` files,
builds rating matrices and constructs the evaluation dataframes (with
negative samples) used by the trainer.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from config import TrainingConfig


__all__ = ["MovieLensDatasetLoader"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Column-name constants                                                   #
# ---------------------------------------------------------------------- #
COL_GROUP_ID = "GroupID"
COL_MOVIE_ID = "MovieID"
COL_RATING = "Rating"
COL_TIMESTAMP = "Timestamp"
COL_MEMBERS = "Members"

VALID_MODES: Tuple[str, ...] = ("user", "group")
VALID_SPLITS: Tuple[str, ...] = ("train", "val", "test")
EVAL_SPLITS: Tuple[str, ...] = ("val", "test")


class MovieLensDatasetLoader:
    """Reader / pre-processor for the MovieLens group-recommendation files."""

    # ================================================================== #
    def __init__(self, config: TrainingConfig) -> None:
        """Pre-load cardinalities and build group bookkeeping.

        Args:
            config: Training configuration whose ``item_num``, ``user_num``,
                ``group_num`` and ``total_group_num`` are filled in here.
        """
        self.config = config
        self.history_length: int = config.history_length

        self.item_num: int = self._count_items()
        self.user_num: int = self._count_users()
        (
            self.group_num,
            self.total_group_num,
            self.group_to_members,
            self.user_to_group_id,
        ) = self._build_group_mappings()

        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ================================================================== #
    # Cardinalities                                                       #
    # ================================================================== #
    def _count_items(self) -> int:
        """Largest item id in ``movies.dat``."""
        df = pd.read_csv(
            self.config.item_path,
            sep="::",
            index_col=0,
            engine="python",
            encoding="iso-8859-1",
        )
        item_num = int(df.index.max())
        self.config.item_num = item_num
        return item_num

    def _count_users(self) -> int:
        """Largest user id in ``users.dat``."""
        df = pd.read_csv(self.config.user_path, sep="::", index_col=0, engine="python")
        user_num = int(df.index.max())
        self.config.user_num = user_num
        return user_num

    # ================================================================== #
    # Groups                                                              #
    # ================================================================== #
    def _build_group_mappings(self) -> Tuple[int, int, Dict[int, tuple], Dict[int, int]]:
        """Build ``group → members`` and ``user → group_id`` lookups.

        Synthetic singleton "user-only groups" are appended after the real
        group list so that user-mode evaluation can reuse the same code
        paths as group mode.
        """
        df_group = pd.read_csv(
            self.config.group_path,
            sep=" ",
            header=None,
            index_col=None,
            names=[COL_GROUP_ID, COL_MEMBERS],
        )
        df_group[COL_MEMBERS] = df_group[COL_MEMBERS].apply(
            lambda raw: tuple(int(uid) for uid in raw.split(","))
        )
        group_num = int(df_group[COL_GROUP_ID].max())

        unique_users = sorted({uid for members in df_group[COL_MEMBERS] for uid in members})
        total_group_num = group_num + len(unique_users)

        df_user_groups = pd.DataFrame(
            {
                COL_GROUP_ID: range(group_num + 1, total_group_num + 1),
                COL_MEMBERS: [(uid,) for uid in unique_users],
            }
        )
        df_combined = pd.concat([df_group, df_user_groups], ignore_index=True)

        group_to_members: Dict[int, tuple] = {
            int(row[COL_GROUP_ID]): row[COL_MEMBERS] for _, row in df_combined.iterrows()
        }
        user_to_group_id: Dict[int, int] = {
            uid: group_num + idx + 1 for idx, uid in enumerate(unique_users)
        }

        self.config.group_num = group_num
        self.config.total_group_num = total_group_num
        return group_num, total_group_num, group_to_members, user_to_group_id

    # ================================================================== #
    # Raw rating loading                                                  #
    # ================================================================== #
    def load_ratings(
        self,
        mode: str,
        dataset_name: str,
        is_appended: bool = True,
    ) -> pd.DataFrame:
        """Load a rating split, optionally concatenated with prior splits.

        Args:
            mode: ``'user'`` or ``'group'``.
            dataset_name: ``'train'``, ``'val'`` or ``'test'``.
            is_appended: If ``True`` (default) ``val`` includes ``train``
                rows and ``test`` includes ``train + val`` rows.

        Returns:
            A pandas dataframe with columns ``GroupID``, ``MovieID``,
            ``Rating``, ``Timestamp``.
        """
        self._check_mode_split(mode, dataset_name, splits=VALID_SPLITS)

        rating_path = (
            Path(self.config.dataset_dir)
            / f"{mode}Rating{dataset_name.capitalize()}.dat"
        )
        df_split = pd.read_csv(
            rating_path,
            sep=" ",
            header=None,
            index_col=None,
            names=[COL_GROUP_ID, COL_MOVIE_ID, COL_RATING, COL_TIMESTAMP],
        )
        logger.info("Read data: %s", rating_path)

        if not is_appended or dataset_name == "train":
            return df_split

        prior_split = "train" if dataset_name == "val" else "val"
        prior = self.load_ratings(mode=mode, dataset_name=prior_split)
        return pd.concat([prior, df_split], ignore_index=True)

    # ================================================================== #
    # Sparse rating-matrix construction                                   #
    # ================================================================== #
    def _build_sparse_matrix(self, df_rating: pd.DataFrame) -> csr_matrix:
        """Materialise the sparse ``(group, item) → rating`` matrix."""
        rows = df_rating[COL_GROUP_ID].to_numpy()
        cols = df_rating[COL_MOVIE_ID].to_numpy()
        values = df_rating[COL_RATING].to_numpy()
        shape = (self.total_group_num + 1, self.config.item_num + 1)
        return coo_matrix((values, (rows, cols)), shape=shape).tocsr()

    def build_rating_matrix(self, dataset_name: str) -> csr_matrix:
        """Build the combined (user-as-group + group) rating matrix."""
        if dataset_name not in VALID_SPLITS:
            raise ValueError(f"dataset_name must be one of {VALID_SPLITS}, got {dataset_name!r}")

        df_user = self._map_user_ids_to_group_ids(self.load_ratings("user", dataset_name))
        df_group = self.load_ratings("group", dataset_name)
        df_all = pd.concat([df_group, df_user], ignore_index=True)
        return self._build_sparse_matrix(df_all)

    def _map_user_ids_to_group_ids(self, df_user_rating: pd.DataFrame) -> pd.DataFrame:
        """Replace user ids with their synthetic singleton-group ids."""
        df_user_rating[COL_GROUP_ID] = df_user_rating[COL_GROUP_ID].map(self.user_to_group_id)
        return df_user_rating

    # ================================================================== #
    # Negative sampling                                                   #
    # ================================================================== #
    def _load_negative_samples(self, mode: str, dataset_name: str) -> Dict[Tuple[int, int], List[int]]:
        """Parse the ``*Negative.dat`` file for the given split."""
        self._check_mode_split(mode, dataset_name, splits=EVAL_SPLITS)
        path = (
            Path(self.config.dataset_dir)
            / f"{mode}Rating{dataset_name.capitalize()}Negative.dat"
        )

        result: Dict[Tuple[int, int], List[int]] = {}
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                tokens = line.split()
                if not tokens:
                    continue
                head = tokens[0][1:-1].split(",")
                group_id = int(head[0])
                if mode == "user":
                    group_id = self.user_to_group_id[group_id]
                item_id = int(head[1])
                result[(group_id, item_id)] = list(map(int, tokens[1:]))
        return result

    # ================================================================== #
    # Evaluation dataframe construction                                   #
    # ================================================================== #
    def _construct_eval_dataframe(
        self,
        df_train: pd.DataFrame,
        df_eval: pd.DataFrame,
        negatives: Dict[Tuple[int, int], List[int]],
    ) -> pd.DataFrame:
        """Build the per-action evaluation dataframe used by the agent."""
        last_state: Dict[int, List[int]] = defaultdict(list)
        groups: List[int] = []
        histories: List[List[int]] = []
        actions: List[int] = []
        negs: List[List[int]] = []

        for group_id, rows in df_train.groupby([COL_GROUP_ID]):
            rows = rows.sort_values(COL_TIMESTAMP, ascending=True, ignore_index=True)
            positives = rows[rows[COL_RATING] == 1][COL_MOVIE_ID].tolist()
            last_state[group_id] = positives[-self.history_length:]

        for group_id, rows in df_eval.groupby([COL_GROUP_ID]):
            group_id = group_id[0]  # tuple → int
            rows = rows.sort_values(COL_TIMESTAMP, ascending=True, ignore_index=True)
            positives = rows[rows[COL_RATING] == 1][COL_MOVIE_ID].tolist()

            window: Deque[int] = deque(maxlen=self.history_length)
            window.extend(last_state[group_id])

            for item_id in positives:
                if len(window) == self.history_length:
                    groups.append(group_id)
                    histories.append(list(window))
                    actions.append(item_id)
                    negs.append(negatives[(group_id, item_id)])
                window.append(item_id)

        return pd.DataFrame(
            {
                "group": groups,
                "history": histories,
                "action": actions,
                "negative samples": negs,
            }
        )

    # ================================================================== #
    # Public evaluation API                                               #
    # ================================================================== #
    def get_evaluation_data(
        self,
        mode: str,
        dataset_name: str,
        reload: bool = False,
    ) -> pd.DataFrame:
        """Load (or build & cache) the evaluation dataframe for ``dataset_name``.

        Args:
            mode: ``'user'`` or ``'group'``.
            dataset_name: ``'val'`` or ``'test'``.
            reload: When ``True`` rebuild even if a cache exists.

        Returns:
            A pandas dataframe ready for the evaluator/trainer.
        """
        self._check_mode_split(mode, dataset_name, splits=EVAL_SPLITS)

        cache_path = (
            Path(self.config.checkpoint_dir)
            / f"eval_{mode}_{dataset_name}_{self.config.history_length}.pkl"
        )

        if cache_path.exists() and not reload:
            df_eval = pd.read_pickle(cache_path)
            logger.info("Load data: %s", cache_path)
            return df_eval

        prior_split = "train" if dataset_name == "val" else "val"
        df_train = self.load_ratings(mode=mode, dataset_name=prior_split)
        df_eval = self.load_ratings(mode=mode, dataset_name=dataset_name, is_appended=False)

        if mode == "user":
            df_train = self._map_user_ids_to_group_ids(df_train)
            df_eval = self._map_user_ids_to_group_ids(df_eval)

        negatives = self._load_negative_samples(mode=mode, dataset_name=dataset_name)
        df_result = self._construct_eval_dataframe(df_train, df_eval, negatives)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df_result.to_pickle(cache_path)
        logger.info("Save data: %s", cache_path)
        return df_result

    # ================================================================== #
    # Helpers                                                             #
    # ================================================================== #
    @staticmethod
    def _check_mode_split(mode: str, dataset_name: str, *, splits: Tuple[str, ...]) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        if dataset_name not in splits:
            raise ValueError(f"dataset_name must be one of {splits}, got {dataset_name!r}")






