"""Training loop for the DDPG group recommender.

The :class:`Trainer` encapsulates warm-up, training, evaluation and
checkpointing. The main loop in :meth:`train` is decomposed into
``_warmup``, ``_run_episode`` and ``_evaluate_and_checkpoint`` helpers.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from agent.ddpg import DDPGRecommenderAgent
from config import TrainingConfig
from environment.env import RecommendationEnvironment
from environment.evaluator import EvaluationResult, TopKMetricsEvaluator
from visualization.plot import plot_losses


__all__ = ["Trainer"]

logger = logging.getLogger(__name__)


_NOISE_FLOOR: float = 0.05


class Trainer:
    """Wrap warm-up, training, evaluation and checkpointing into one object.

    Improvements over the baseline DDPG loop:

    * Replay buffer is filled with random transitions before any update.
    * Multiple gradient updates per environment step.
    * Cosine-annealing LR schedules stepped once per episode.
    * Linear OU-noise ε decay for smoother exploration → exploitation.
    * Best-model checkpointing based on Group Recall@10.
    """

    # ================================================================== #
    def __init__(
        self,
        config: TrainingConfig,
        env: RecommendationEnvironment,
        agent: DDPGRecommenderAgent,
        evaluator: TopKMetricsEvaluator,
        df_eval_user: pd.DataFrame,
        df_eval_group: pd.DataFrame,
    ) -> None:
        self.config = config
        self.env = env
        self.agent = agent
        self.evaluator = evaluator
        self.user_eval_df = df_eval_user
        self.group_eval_df = df_eval_group

    # ================================================================== #
    # Public API                                                          #
    # ================================================================== #
    def train(self) -> None:
        """Run the full training loop."""
        cfg = self.config

        self._warmup()

        train_actor_losses: List[Optional[float]] = []
        train_critic_losses: List[Optional[float]] = []
        eval_actor_losses: List[float] = []
        eval_critic_losses: List[float] = []
        eval_episode_indices: List[int] = []

        best_group_recall = -1.0
        best_episode = -1

        noise_eps_start, noise_eps_end = 1.0, _NOISE_FLOOR
        noise_decay = (noise_eps_start - noise_eps_end) / max(cfg.max_episodes - 1, 1)
        noise_eps = noise_eps_start

        for episode in range(cfg.max_episodes):
            mean_reward, actor_loss, critic_loss = self._run_episode(noise_eps=noise_eps)

            self.agent.embedding_scheduler.step()
            self.agent.actor_scheduler.step()
            self.agent.critic_scheduler.step()

            noise_eps = max(noise_eps_end, noise_eps - noise_decay)
            self.agent.noise.sigma = cfg.ou_sigma * noise_eps

            train_actor_losses.append(actor_loss)
            train_critic_losses.append(critic_loss)
            logger.info("Episode = %d, average reward = %.4f", episode, mean_reward)

            if (episode + 1) % cfg.eval_interval_episodes == 0:
                best_group_recall, best_episode = self._evaluate_and_checkpoint(
                    episode=episode,
                    best_group_recall=best_group_recall,
                    best_episode=best_episode,
                    eval_actor_losses=eval_actor_losses,
                    eval_critic_losses=eval_critic_losses,
                    eval_episode_indices=eval_episode_indices,
                )

        logger.info(
            "Training complete. Best Group Recall@10 = %.4f at episode %d.",
            best_group_recall, best_episode,
        )

        plot_losses(
            train_actor_losses=train_actor_losses,
            train_critic_losses=train_critic_losses,
            eval_actor_losses=eval_actor_losses,
            eval_critic_losses=eval_critic_losses,
            eval_loss_steps=eval_episode_indices,
            save_dir=cfg.output_dir,
        )

    # ================================================================== #
    # Phases                                                              #
    # ================================================================== #
    def _warmup(self) -> None:
        """Fill the replay buffer with random transitions."""
        cfg = self.config
        logger.info("Warm-up: collecting %d random transitions ...", cfg.warmup_steps)

        state = self.env.reset()
        for _ in range(cfg.warmup_steps):
            action = int(np.random.randint(1, cfg.item_num + 1))
            new_state, reward, _, _ = self.env.step(action)
            shaped = self.agent.compute_shaped_reward(
                raw_reward=reward, action=action, history=state[1:]
            )
            self.agent.replay_buffer.add((state, action, shaped, new_state))
            state = new_state

        logger.info("Warm-up complete. Buffer size: %d", len(self.agent.replay_buffer))

    # ------------------------------------------------------------------ #
    def _run_episode(self, *, noise_eps: float) -> Tuple[float, Optional[float], Optional[float]]:
        """Run a single training episode.

        Args:
            noise_eps: Current OU exploration ε.

        Returns:
            ``(mean_reward, mean_actor_loss, mean_critic_loss)``.
        """
        cfg = self.config
        agent = self.agent
        env = self.env

        state = env.reset()
        agent.noise.reset()
        episode_reward = 0
        actor_losses: List[float] = []
        critic_losses: List[float] = []
        explore = noise_eps > _NOISE_FLOOR

        for _ in range(cfg.steps_per_episode):
            with torch.no_grad():
                action = agent.select_action(state, with_noise=explore)

            new_state, reward, _, _ = env.step(action)
            shaped = agent.compute_shaped_reward(
                raw_reward=reward, action=action, history=state[1:]
            )
            agent.replay_buffer.add((state, action, shaped, new_state))
            state = new_state
            episode_reward += reward

            if len(agent.replay_buffer) >= cfg.batch_size:
                for _ in range(cfg.gradient_steps_per_env_step):
                    a_loss, c_loss = agent.update()
                    actor_losses.append(a_loss)
                    critic_losses.append(c_loss)

        mean_reward = episode_reward / cfg.steps_per_episode
        mean_actor = (sum(actor_losses) / len(actor_losses)) if actor_losses else None
        mean_critic = (sum(critic_losses) / len(critic_losses)) if critic_losses else None
        return mean_reward, mean_actor, mean_critic

    # ------------------------------------------------------------------ #
    def _evaluate_and_checkpoint(
        self,
        episode: int,
        best_group_recall: float,
        best_episode: int,
        eval_actor_losses: List[float],
        eval_critic_losses: List[float],
        eval_episode_indices: List[int],
    ) -> Tuple[float, int]:
        """Run validation, persist losses, and checkpoint when improved."""
        cfg = self.config

        actor_loss, critic_loss = self.agent.evaluate_loss(df_eval=self.user_eval_df)
        if actor_loss is not None and critic_loss is not None:
            eval_actor_losses.append(actor_loss)
            eval_critic_losses.append(critic_loss)
            eval_episode_indices.append(episode)
            logger.info(
                "Eval losses at episode %d: actor=%.4f, critic=%.4f",
                episode, actor_loss, critic_loss,
            )

        for top_k in cfg.eval_top_k_values:
            self.evaluator.evaluate(
                agent=self.agent, df_eval=self.user_eval_df, mode="user", top_K=top_k
            )

        group_metrics: Dict[int, EvaluationResult] = {}
        for top_k in cfg.eval_top_k_values:
            group_metrics[top_k] = self.evaluator.evaluate(
                agent=self.agent, df_eval=self.group_eval_df, mode="group", top_K=top_k
            )

        recall_at_10 = group_metrics.get(10).recall if 10 in group_metrics else 0.0
        if recall_at_10 > best_group_recall:
            best_group_recall = recall_at_10
            best_episode = episode
            self._save_best_model()
            logger.info(
                "  *** New best model saved (episode %d, Group Recall@10 = %.4f) ***",
                episode, best_group_recall,
            )

        return best_group_recall, best_episode

    # ================================================================== #
    # Checkpointing                                                       #
    # ================================================================== #
    def _save_best_model(self) -> None:
        """Persist the best model state-dicts to disk.

        The on-disk filenames (``best_embedding.pt``, ``best_actor.pt``,
        ``best_critic.pt``) are intentionally preserved.
        """
        save_dir = Path(self.config.checkpoint_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.agent.embedding.state_dict(), save_dir / "best_embedding.pt")
        torch.save(self.agent.actor.state_dict(), save_dir / "best_actor.pt")
        torch.save(self.agent.critic.state_dict(), save_dir / "best_critic.pt")



