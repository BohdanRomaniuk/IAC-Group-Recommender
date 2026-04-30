"""Project entry point.

Wires every component together (config → data → environment → agent →
trainer) and starts a training run. Logging is configured here so that the
modules' loggers emit to stdout when the application is launched directly.
"""
from __future__ import annotations

import argparse
import logging

from agent.ddpg import DDPGRecommenderAgent
from config import TrainingConfig
from environment.dataloader import MovieLensDatasetLoader
from environment.env import RecommendationEnvironment
from environment.evaluator import TopKMetricsEvaluator
from environment.trainer import Trainer
from utils.noise import OrnsteinUhlenbeckNoise


def _configure_logging(level: int = logging.INFO) -> None:
    """Set up a sane root-logger format for the CLI."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IAC Group Recommender — DDPG training")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a YAML config file (e.g. default.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    """Build all components and launch training."""
    _configure_logging()
    args = _parse_args()

    config = TrainingConfig.from_yaml(args.config) if args.config else TrainingConfig()

    dataset = MovieLensDatasetLoader(config)
    rating_matrix_train = dataset.build_rating_matrix(dataset_name="val")
    user_eval_df = dataset.get_evaluation_data(mode="user", dataset_name="test")
    group_eval_df = dataset.get_evaluation_data(mode="group", dataset_name="test")

    env = RecommendationEnvironment(
        config=config, rating_matrix=rating_matrix_train, dataset_name="val"
    )
    noise = OrnsteinUhlenbeckNoise(config=config)
    agent = DDPGRecommenderAgent(
        config=config,
        noise=noise,
        group_to_members=dataset.group_to_members,
        verbose=True,
    )
    evaluator = TopKMetricsEvaluator(config=config)

    trainer = Trainer(
        config=config,
        env=env,
        agent=agent,
        evaluator=evaluator,
        df_eval_user=user_eval_df,
        df_eval_group=group_eval_df,
    )
    trainer.train()


if __name__ == "__main__":
    main()

