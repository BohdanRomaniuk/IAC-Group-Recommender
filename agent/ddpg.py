"""
DDPG (Deep Deterministic Policy Gradient) Agent.
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_optimizer import Ranger

from agent.replay_buffer import PrioritizedReplayBuffer
from config import Config
from models.actor import Actor
from models.critic import Critic
from models.integrator import Embedding
from utils.noise import OUNoise


class DDPGAgent(object):
    """
    DDPG Agent with prioritized replay and entropy-regularised actor.
    """

    def __init__(self, config: Config, noise: OUNoise, group2members_dict: dict, verbose: bool = False):
        """
        Initialize DDPGAgent

        :param config: configurations
        :param noise: OU noise instance
        :param group2members_dict: group members data
        :param verbose: True to print networks
        """
        self.config = config
        self.noise = noise
        self.group2members_dict = group2members_dict
        self.tau = config.tau
        self.gamma = config.gamma
        self.device = config.device

        self.embedding = Embedding(embedding_size=config.embedding_size,
                                   user_num=config.user_num,
                                   item_num=config.item_num).to(config.device)
        self.actor = Actor(embedded_state_size=config.embedded_state_size,
                           action_weight_size=config.embedded_action_size,
                           hidden_sizes=config.actor_hidden_sizes).to(config.device)
        self.actor_target = Actor(embedded_state_size=config.embedded_state_size,
                                  action_weight_size=config.embedded_action_size,
                                  hidden_sizes=config.actor_hidden_sizes).to(config.device)
        self.critic = Critic(embedded_state_size=config.embedded_state_size,
                             embedded_action_size=config.embedded_action_size,
                             hidden_sizes=config.critic_hidden_sizes).to(config.device)
        self.critic_target = Critic(embedded_state_size=config.embedded_state_size,
                                    embedded_action_size=config.embedded_action_size,
                                    hidden_sizes=config.critic_hidden_sizes).to(config.device)

        if verbose:
            print(self.embedding)
            print(self.actor)
            print(self.critic)

        self.copy_network(self.actor, self.actor_target)
        self.copy_network(self.critic, self.critic_target)

        self.replay_memory = PrioritizedReplayBuffer(
            buffer_size=config.buffer_size,
            alpha=config.per_alpha,
            beta_start=config.per_beta_start,
            beta_frames=config.per_beta_frames,
        )

        self.critic_criterion = nn.MSELoss(reduction='none')
        self.embedding_optimizer = Ranger(self.embedding.parameters(), lr=config.embedding_learning_rate,
                                          weight_decay=config.embedding_weight_decay)
        self.actor_optimizer = Ranger(self.actor.parameters(), lr=config.actor_learning_rate,
                                      weight_decay=config.actor_weight_decay)
        self.critic_optimizer = Ranger(self.critic.parameters(), lr=config.critic_learning_rate,
                                       weight_decay=config.critic_weight_decay)

        T_max = config.num_episodes
        self.embedding_scheduler = CosineAnnealingLR(self.embedding_optimizer, T_max=T_max, eta_min=1e-5)
        self.actor_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=T_max, eta_min=1e-5)
        self.critic_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=T_max, eta_min=1e-5)

    # ------------------------------------------------------------------ #
    #  Reward shaping                                                      #
    # ------------------------------------------------------------------ #

    def shaped_reward(self, raw_reward: float, action: int, history: list) -> float:
        """
        Augment the environment reward with diversity and coverage bonuses.
        """
        alpha = self.config.reward_diversity_alpha
        beta = self.config.reward_coverage_beta

        diversity = 1.0 if action not in history else 0.0
        coverage = 1.0

        return raw_reward + alpha * diversity + beta * coverage

    # ------------------------------------------------------------------ #

    def copy_network(self, network: nn.Module, network_target: nn.Module) -> None:
        """Copy one network to its target network."""
        for parameters, target_parameters in zip(network.parameters(), network_target.parameters()):
            target_parameters.data.copy_(parameters.data)

    def sync_network(self, network: nn.Module, network_target: nn.Module) -> None:
        """Soft-update target network weights."""
        for parameters, target_parameters in zip(network.parameters(), network_target.parameters()):
            target_parameters.data.copy_(parameters.data * self.tau + target_parameters.data * (1 - self.tau))

    def get_action(self, state: list, item_candidates: Optional[list] = None,
                   top_K: int = 1, with_noise: bool = False):
        """Get one action."""
        with torch.no_grad():
            states = [state]
            embedded_states = self.embed_states(states)
            action_weights = self.actor(embedded_states)
            action_weight = torch.squeeze(action_weights)
            if with_noise:
                action_weight = action_weight + self.noise.get_ou_noise().to(action_weight.device)

            if item_candidates is None:
                item_embedding_weight = self.embedding.item_embedding.weight.clone()
            else:
                item_candidates = np.array(item_candidates)
                item_candidates_tensor = torch.tensor(item_candidates, dtype=torch.int).to(self.device)
                item_embedding_weight = self.embedding.item_embedding(item_candidates_tensor)

            scores = torch.inner(action_weight, item_embedding_weight).detach().cpu().numpy()
            sorted_score_indices = np.argsort(scores)[::-1][:top_K].copy()

            if item_candidates is None:
                action = sorted_score_indices
            else:
                action = item_candidates[sorted_score_indices]
            action = np.squeeze(action)
            if top_K == 1:
                action = action.item()
        return action

    def get_embedded_actions(self, embedded_states: torch.Tensor, target: bool = False) -> torch.Tensor:
        """Get embedded actions."""
        if not target:
            action_weights = self.actor(embedded_states)
        else:
            action_weights = self.actor_target(embedded_states)

        item_embedding_weight = self.embedding.item_embedding.weight.clone()
        scores = torch.inner(action_weights, item_embedding_weight)
        embedded_actions = torch.inner(functional.gumbel_softmax(scores, hard=True), item_embedding_weight.t())
        return embedded_actions

    def embed_state(self, state: list) -> torch.Tensor:
        """Embed one state."""
        group_id = state[0]
        group_members = torch.tensor(self.group2members_dict[group_id], dtype=torch.int).to(self.device)
        history = torch.tensor(state[1:], dtype=torch.int).to(self.device)
        embedded_state = self.embedding(group_members, history)
        return embedded_state

    def embed_states(self, states: List[list]) -> torch.Tensor:
        """Embed states."""
        embedded_states = torch.stack([self.embed_state(state) for state in states], dim=0)
        return embedded_states

    def embed_actions(self, actions: list) -> torch.Tensor:
        """Embed actions."""
        actions_t = torch.tensor(actions, dtype=torch.int).to(self.device)
        embedded_actions = self.embedding.item_embedding(actions_t)
        return embedded_actions

    def update(self) -> Tuple[float, float]:
        """
        Update the networks using a prioritized batch.
        Critic loss is weighted by IS weights to correct the sampling bias.
        Actor loss includes an entropy regularisation term.

        :return: actor loss and critic loss
        """
        batch, per_indices, is_weights = self.replay_memory.sample(self.config.batch_size)
        is_weights = is_weights.to(self.device).unsqueeze(-1)  # [B, 1]

        states, actions, rewards, next_states = list(zip(*batch))

        # ---- Critic update -------------------------------------------- #
        self.critic_optimizer.zero_grad()

        embedded_states = self.embed_states(states)
        embedded_actions = self.embed_actions(actions)
        rewards_t = torch.unsqueeze(torch.tensor(rewards, dtype=torch.float).to(self.device), dim=-1)

        with torch.no_grad():
            embedded_next_states = self.embed_states(next_states)
            embedded_next_actions = self.get_embedded_actions(embedded_next_states, target=True)
            next_q_values = self.critic_target(embedded_next_states, embedded_next_actions)
            q_values_target = rewards_t + self.gamma * next_q_values

        q_values = self.critic(embedded_states, embedded_actions)

        per_sample_loss = self.critic_criterion(q_values, q_values_target)  # [B, 1]
        critic_loss = (is_weights * per_sample_loss).mean()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        td_errors = (q_values - q_values_target).abs().detach().cpu().numpy().flatten()
        self.replay_memory.update_priorities(per_indices, td_errors)

        # ---- Actor + Embedding update --------------------------------- #
        self.actor_optimizer.zero_grad()
        self.embedding_optimizer.zero_grad()

        embedded_states = self.embed_states(states)

        action_weights = self.actor(embedded_states)
        item_emb = self.embedding.item_embedding.weight
        scores = torch.inner(action_weights, item_emb)
        log_probs = torch.log_softmax(scores, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        embedded_actions_actor = self.get_embedded_actions(embedded_states)
        q_for_actor = self.critic(embedded_states, embedded_actions_actor).mean()

        actor_loss = -q_for_actor - self.config.entropy_coef * entropy
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.embedding.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        self.embedding_optimizer.step()

        self.sync_network(self.actor, self.actor_target)
        self.sync_network(self.critic, self.critic_target)

        actor_loss_val = float(actor_loss.detach().cpu().numpy())
        critic_loss_val = float(critic_loss.detach().cpu().numpy())

        return actor_loss_val, critic_loss_val

    def evaluate_loss(self, df_eval: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute actor and critic losses on held-out evaluation data
        without updating any network weights.
        """
        if df_eval is None or (isinstance(df_eval, pd.DataFrame) and df_eval.empty):
            return None, None

        rows = df_eval.sample(n=min(self.config.batch_size, len(df_eval)), replace=False)

        states, actions, next_states = [], [], []
        for _, row in rows.iterrows():
            group = row['group']
            history = list(row['history'])
            action = row['action']

            state = [group] + history
            next_history = history[1:] + [action]
            next_state = [group] + next_history

            states.append(state)
            actions.append(action)
            next_states.append(next_state)

        with torch.no_grad():
            embedded_states = self.embed_states(states)
            embedded_actions = self.embed_actions(actions)
            rewards_t = torch.ones(len(states), 1, dtype=torch.float).to(self.device)
            embedded_next_states = self.embed_states(next_states)

            q_values = self.critic(embedded_states, embedded_actions)
            embedded_next_actions = self.get_embedded_actions(embedded_next_states, target=True)
            next_q_values = self.critic_target(embedded_next_states, embedded_next_actions)
            q_values_target = rewards_t + self.gamma * next_q_values

            eval_criterion = nn.MSELoss()
            critic_loss = float(eval_criterion(q_values, q_values_target).cpu().numpy())

            action_weights = self.actor(embedded_states)
            item_emb = self.embedding.item_embedding.weight.clone()
            scores = torch.inner(action_weights, item_emb)
            log_probs = torch.log_softmax(scores, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            embedded_actions_actor = self.get_embedded_actions(embedded_states)
            q_for_actor = self.critic(embedded_states, embedded_actions_actor).mean()
            actor_loss = float((-q_for_actor - self.config.entropy_coef * entropy).cpu().numpy())

        return actor_loss, critic_loss

