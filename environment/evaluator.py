"""
Evaluator: Recall@K, NDCG@K, Precision@K, MAP, HR, MRR and Coverage.
"""
from typing import Tuple

import numpy as np
import pandas as pd

from agent.ddpg import DDPGAgent
from config import Config


class Evaluator(object):
    """
    Evaluate top-K recommendation performance against negative samples.
    """

    def __init__(self, config: Config):
        """
        Initialize Evaluator

        :param config: configurations
        """
        self.config = config

    def evaluate(self, agent: DDPGAgent, df_eval: pd.DataFrame, mode: str,
                 top_K: int = 5) -> Tuple[float, float, float, float, float, float, float]:
        """
        Evaluate the agent.

        :param agent: agent
        :param df_eval: evaluation data
        :param mode: in ['user', 'group']
        :param top_K: length of the recommendation list
        :return: (recall, ndcg, precision, map, hit_rate, mrr, coverage)
        """
        recall_scores = []
        ndcg_scores = []
        precision_scores = []
        map_scores = []
        hit_rate_scores = []
        mrr_scores = []
        all_recommended_items = set()
        all_candidate_items = set()

        for _, row in df_eval.iterrows():
            group = row['group']
            history = row['history']
            item_true = row['action']
            item_candidates = row['negative samples'] + [item_true]
            np.random.shuffle(item_candidates)

            all_candidate_items.update(item_candidates)

            state = [group] + history
            items_pred = agent.get_action(state=state, item_candidates=item_candidates, top_K=top_K)

            all_recommended_items.update(items_pred)

            recall_score = 0
            ndcg_score = 0
            precision_score = 0
            map_score = 0
            hit_rate_score = 0
            mrr_score = 0

            for k, item in enumerate(items_pred):
                if item == item_true:
                    recall_score = 1
                    ndcg_score = np.log2(2) / np.log2(k + 2)
                    precision_score = 1 / top_K
                    map_score = 1 / (k + 1)
                    hit_rate_score = 1
                    mrr_score = 1 / (k + 1)
                    break

            recall_scores.append(recall_score)
            ndcg_scores.append(ndcg_score)
            precision_scores.append(precision_score)
            map_scores.append(map_score)
            hit_rate_scores.append(hit_rate_score)
            mrr_scores.append(mrr_score)

        avg_recall_score = float(np.mean(recall_scores))
        avg_ndcg_score = float(np.mean(ndcg_scores))
        avg_precision_score = float(np.mean(precision_scores))
        avg_map_score = float(np.mean(map_scores))
        avg_hit_rate_score = float(np.mean(hit_rate_scores))
        avg_mrr_score = float(np.mean(mrr_scores))
        coverage = float(len(all_recommended_items) / len(all_candidate_items)) if all_candidate_items else 0.0
        print('%s: Recall@%d = %.4f, NDCG@%d = %.4f, Precision@%d = %.4f, MAP@%d = %.4f, HR@%d = %.4f, MRR@%d = %.4f, Coverage@%d = %.4f' % (
            mode.capitalize(),
            top_K, avg_recall_score,
            top_K, avg_ndcg_score,
            top_K, avg_precision_score,
            top_K, avg_map_score,
            top_K, avg_hit_rate_score,
            top_K, avg_mrr_score,
            top_K, coverage
        ))
        return avg_recall_score, avg_ndcg_score, avg_precision_score, avg_map_score, avg_hit_rate_score, avg_mrr_score, coverage

