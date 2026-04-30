"""
Trainer: encapsulates the full DDPG training loop.
"""
import os

import numpy as np
import pandas as pd
import torch

from agent.ddpg import DDPGAgent
from config import Config
from environment.env import Env
from environment.evaluator import Evaluator
from visualization.plot import plot_losses


class Trainer(object):
    """
    Trainer class. Encapsulates warm-up, training, evaluation, checkpointing
    and loss-curve plotting.
    """

    def __init__(self, config: Config, env: Env, agent: DDPGAgent, evaluator: Evaluator,
                 df_eval_user: pd.DataFrame, df_eval_group: pd.DataFrame):
        """
        Initialize Trainer

        :param config: configurations
        :param env: environment
        :param agent: DDPG agent
        :param evaluator: evaluator
        :param df_eval_user: user evaluation data
        :param df_eval_group: group evaluation data
        """
        self.config = config
        self.env = env
        self.agent = agent
        self.evaluator = evaluator
        self.df_eval_user = df_eval_user
        self.df_eval_group = df_eval_group

    # ------------------------------------------------------------------ #
    def _save_best_model(self) -> None:
        """Save the best model state dicts to disk."""
        os.makedirs(self.config.saves_folder_path, exist_ok=True)
        torch.save(self.agent.embedding.state_dict(),
                   os.path.join(self.config.saves_folder_path, 'best_embedding.pt'))
        torch.save(self.agent.actor.state_dict(),
                   os.path.join(self.config.saves_folder_path, 'best_actor.pt'))
        torch.save(self.agent.critic.state_dict(),
                   os.path.join(self.config.saves_folder_path, 'best_critic.pt'))

    # ------------------------------------------------------------------ #
    def train(self) -> None:
        """
        Train the agent with the environment.

        Improvements over the baseline:
          - Warmup phase: fill replay buffer with random actions before any gradient update.
          - Multiple gradient updates per environment step.
          - Cosine-annealing LR scheduling (stepped once per episode).
          - OU-noise epsilon decay for smoother exploration→exploitation transition.
          - Best-model checkpointing based on Group Recall@10.
        """
        config = self.config
        env = self.env
        agent = self.agent
        evaluator = self.evaluator
        df_eval_user = self.df_eval_user
        df_eval_group = self.df_eval_group

        rewards = []
        train_actor_losses_per_episode = []
        train_critic_losses_per_episode = []
        eval_actor_losses_per_episode = []
        eval_critic_losses_per_episode = []
        eval_episode_indices = []

        best_group_recall = -1.0
        best_episode = -1

        # ------------------- Warm-up ---------------------------------- #
        print('-' * 50)
        print(f'Warm-up: collecting {config.warmup_steps} random transitions …')
        warmup_state = env.reset()
        for _ in range(config.warmup_steps):
            random_action = np.random.randint(1, config.item_num + 1)
            new_state, reward, _, _ = env.step(random_action)
            shaped = agent.shaped_reward(raw_reward=reward, action=random_action, history=warmup_state[1:])
            agent.replay_memory.push((warmup_state, random_action, shaped, new_state))
            warmup_state = new_state
        print(f'Warm-up complete. Buffer size: {len(agent.replay_memory)}')
        print('-' * 50)

        # Noise scale decays linearly from 1.0 → 0.05 over all episodes
        noise_epsilon_start = 1.0
        noise_epsilon_end = 0.05
        noise_decay = (noise_epsilon_start - noise_epsilon_end) / max(config.num_episodes - 1, 1)
        noise_epsilon = noise_epsilon_start

        for episode in range(config.num_episodes):
            state = env.reset()
            agent.noise.reset()
            episode_reward = 0
            episode_actor_losses = []
            episode_critic_losses = []

            for step in range(config.num_steps):
                with torch.no_grad():
                    action = agent.get_action(state, with_noise=(noise_epsilon > 0.05))

                new_state, reward, _, _ = env.step(action)
                shaped = agent.shaped_reward(raw_reward=reward, action=action, history=state[1:])
                agent.replay_memory.push((state, action, shaped, new_state))
                state = new_state
                episode_reward += reward

                if len(agent.replay_memory) >= config.batch_size:
                    for _ in range(config.num_updates_per_step):
                        actor_loss, critic_loss = agent.update()
                        episode_actor_losses.append(actor_loss)
                        episode_critic_losses.append(critic_loss)

            agent.embedding_scheduler.step()
            agent.actor_scheduler.step()
            agent.critic_scheduler.step()

            noise_epsilon = max(noise_epsilon_end, noise_epsilon - noise_decay)
            agent.noise.ou_sigma = config.ou_sigma * noise_epsilon

            if episode_actor_losses:
                train_actor_losses_per_episode.append(sum(episode_actor_losses) / len(episode_actor_losses))
                train_critic_losses_per_episode.append(sum(episode_critic_losses) / len(episode_critic_losses))
            else:
                train_actor_losses_per_episode.append(None)
                train_critic_losses_per_episode.append(None)

            rewards.append(episode_reward / config.num_steps)
            print('Episode = %d, average reward = %.4f' % (episode, episode_reward / config.num_steps))

            if (episode + 1) % config.eval_per_iter == 0:
                eval_actor_loss, eval_critic_loss = agent.evaluate_loss(df_eval=df_eval_user)
                if eval_actor_loss is not None:
                    eval_actor_losses_per_episode.append(eval_actor_loss)
                    eval_critic_losses_per_episode.append(eval_critic_loss)
                    eval_episode_indices.append(episode)
                    print('Eval losses at episode %d: actor=%.4f, critic=%.4f' % (
                        episode, eval_actor_loss, eval_critic_loss))

                for top_K in config.top_K_list:
                    evaluator.evaluate(agent=agent, df_eval=df_eval_user, mode='user', top_K=top_K)

                group_metrics = {}
                for top_K in config.top_K_list:
                    metrics = evaluator.evaluate(agent=agent, df_eval=df_eval_group, mode='group', top_K=top_K)
                    group_metrics[top_K] = metrics

                current_recall_10 = group_metrics.get(10, (0,))[0]
                if current_recall_10 > best_group_recall:
                    best_group_recall = current_recall_10
                    best_episode = episode
                    self._save_best_model()
                    print(f'  *** New best model saved (episode {episode}, '
                          f'Group Recall@10 = {best_group_recall:.4f}) ***')

        print(f'\nTraining complete. Best Group Recall@10 = {best_group_recall:.4f} at episode {best_episode}.')

        plot_losses(
            train_actor_losses=train_actor_losses_per_episode,
            train_critic_losses=train_critic_losses_per_episode,
            eval_actor_losses=eval_actor_losses_per_episode,
            eval_critic_losses=eval_critic_losses_per_episode,
            eval_loss_steps=eval_episode_indices,
            save_dir=self.config.results_folder_path,
        )

