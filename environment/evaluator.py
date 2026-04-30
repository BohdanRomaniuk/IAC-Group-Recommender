"""Top-K evaluation metrics for the recommender.

Provides :class:`TopKMetricsEvaluator` computing Recall@K, NDCG@K,
Precision@K, MAP@K, HR@K, MRR@K and catalogue Coverage.
:class:`EvaluationResult` is a ``NamedTuple`` so it is tuple-unpackable.
"""
from __future__ import annotations

import logging
from typing import NamedTuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from config import TrainingConfig


if TYPE_CHECKING:  # avoid circular import
    from agent.ddpg import DDPGRecommenderAgent


__all__ = ["TopKMetricsEvaluator", "EvaluationResult"]

logger = logging.getLogger(__name__)


_VALID_MODES = ("user", "group")


class EvaluationResult(NamedTuple):
    """Aggregated top-K metrics for a single evaluation pass.

    Tuple order matches the legacy return signature so it can still be
    unpacked as ``recall, ndcg, precision, mAP, hr, mrr, coverage``.
    """

    recall: float
    ndcg: float
    precision: float
    map: float
    hit_rate: float
    mrr: float
    coverage: float


class TopKMetricsEvaluator:
    """Compute top-K ranking metrics against a positive + negatives set."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    # ================================================================== #
    def evaluate(
        self,
        agent: "DDPGRecommenderAgent",
        df_eval: pd.DataFrame,
        mode: str,
        top_K: int = 5,
    ) -> EvaluationResult:
        """Run an evaluation pass and log the aggregated metrics.

        Public name kept stable; new callers may use the keyword
        ``top_k`` interchangeably via positional argument.

        Args:
            agent: The DDPG agent to evaluate.
            df_eval: Evaluation dataframe.
            mode: ``'user'`` or ``'group'`` (used in the log line only).
            top_K: Length of the recommendation list.

        Returns:
            An :class:`EvaluationResult`. The tuple ordering matches the
            legacy 7-tuple so existing unpacking code keeps working.
        """
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")

        recalls: list[float] = []
        ndcgs: list[float] = []
        precisions: list[float] = []
        maps: list[float] = []
        hit_rates: list[float] = []
        mrrs: list[float] = []
        recommended_pool: set[int] = set()
        candidate_pool: set[int] = set()

        for _, row in df_eval.iterrows():
            group = row["group"]
            history = row["history"]
            true_item = row["action"]
            candidates = list(row["negative samples"]) + [true_item]
            np.random.shuffle(candidates)
            candidate_pool.update(candidates)

            state = [group] + list(history)
            predictions = agent.select_action(
                state=state, item_candidates=candidates, top_k=top_K
            )
            recommended_pool.update(predictions)

            recall, ndcg, precision, ap, hr, mrr = self._score_one(predictions, true_item, top_K)
            recalls.append(recall)
            ndcgs.append(ndcg)
            precisions.append(precision)
            maps.append(ap)
            hit_rates.append(hr)
            mrrs.append(mrr)

        coverage = (
            float(len(recommended_pool) / len(candidate_pool)) if candidate_pool else 0.0
        )
        result = EvaluationResult(
            recall=float(np.mean(recalls)),
            ndcg=float(np.mean(ndcgs)),
            precision=float(np.mean(precisions)),
            map=float(np.mean(maps)),
            hit_rate=float(np.mean(hit_rates)),
            mrr=float(np.mean(mrrs)),
            coverage=coverage,
        )

        logger.info(
            "%s: Recall@%d=%.4f, NDCG@%d=%.4f, Precision@%d=%.4f, "
            "MAP@%d=%.4f, HR@%d=%.4f, MRR@%d=%.4f, Coverage@%d=%.4f",
            mode.capitalize(),
            top_K, result.recall,
            top_K, result.ndcg,
            top_K, result.precision,
            top_K, result.map,
            top_K, result.hit_rate,
            top_K, result.mrr,
            top_K, result.coverage,
        )
        return result

    # ================================================================== #
    @staticmethod
    def _score_one(
        predictions,
        true_item: int,
        top_k: int,
    ) -> tuple[float, float, float, float, float, float]:
        """Compute per-row contribution to each metric."""
        for rank, item in enumerate(predictions):
            if item == true_item:
                ndcg = float(np.log2(2) / np.log2(rank + 2))
                return (
                    1.0,                # recall
                    ndcg,               # ndcg
                    1.0 / top_k,        # precision
                    1.0 / (rank + 1),   # average precision (single relevant)
                    1.0,                # hit-rate
                    1.0 / (rank + 1),   # MRR
                )
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)




