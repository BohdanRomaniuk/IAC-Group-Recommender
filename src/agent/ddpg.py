"""DDPG agent for group recommendation.

The :class:`DDPGRecommenderAgent` orchestrates the actor, critic, target
networks, integrator (state encoder), prioritised replay buffer and OU
exploration noise.
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_optimizer import Ranger
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from agent.replay_buffer import PrioritizedExperienceReplay
from config import TrainingConfig
from models.actor import Actor
from models.critic import Critic
from models.integrator import Integrator
from utils.noise import OrnsteinUhlenbeckNoise


__all__ = ["DDPGRecommenderAgent"]

logger = logging.getLogger(__name__)


class DDPGRecommenderAgent:
    """DDPG agent with prioritised replay and entropy-regularised actor."""

    # ================================================================== #
    # Construction                                                        #
    # ================================================================== #
    def __init__(
        self,
        config: TrainingConfig,
        noise: OrnsteinUhlenbeckNoise,
        group_to_members: dict,
        verbose: bool = False,
    ) -> None:
        """Build all networks, optimisers, schedulers and the replay buffer.

        Args:
            config: Training configuration.
            noise: OU noise instance for exploration.
            group_to_members: Mapping ``group_id → tuple(user_ids)``.
            verbose: When ``True``, log the network summaries.
            **legacy_kwargs: Tolerates ``group2members_dict=`` callers.
        """
        self.config = config
        self.noise = noise
        self.group_to_members = group_to_members
        self.tau: float = config.tau
        self.gamma: float = config.gamma
        self.device: torch.device = config.device

        self._build_networks(verbose=verbose)
        self._build_buffer()
        self._build_optimisers()

    # ------------------------------------------------------------------ #
    def _build_networks(self, *, verbose: bool) -> None:
        cfg = self.config
        self.embedding = Integrator(
            embedding_size=cfg.embedding_size,
            user_num=cfg.user_num,
            item_num=cfg.item_num,
        ).to(self.device)

        actor_kwargs = dict(
            embedded_state_size=cfg.state_embedding_dim,
            action_weight_size=cfg.action_embedding_dim,
            hidden_sizes=cfg.actor_hidden_sizes,
        )
        critic_kwargs = dict(
            embedded_state_size=cfg.state_embedding_dim,
            embedded_action_size=cfg.action_embedding_dim,
            hidden_sizes=cfg.critic_hidden_sizes,
        )

        self.actor = Actor(**actor_kwargs).to(self.device)
        self.actor_target = Actor(**actor_kwargs).to(self.device)
        self.critic = Critic(**critic_kwargs).to(self.device)
        self.critic_target = Critic(**critic_kwargs).to(self.device)

        if verbose:
            for module in (self.embedding, self.actor, self.critic):
                logger.info("%s", module)

        self.hard_update_target(self.actor, self.actor_target)
        self.hard_update_target(self.critic, self.critic_target)

    # ------------------------------------------------------------------ #
    def _build_buffer(self) -> None:
        cfg = self.config
        self.replay_buffer = PrioritizedExperienceReplay(
            buffer_size=cfg.buffer_size,
            alpha=cfg.per_alpha,
            beta_start=cfg.per_beta_start,
            beta_frames=cfg.per_beta_frames,
        )

    # ------------------------------------------------------------------ #
    def _build_optimisers(self) -> None:
        cfg = self.config
        self.critic_criterion = nn.MSELoss(reduction="none")

        self.embedding_optimizer = Ranger(
            self.embedding.parameters(),
            lr=cfg.embedding_learning_rate,
            weight_decay=cfg.embedding_weight_decay,
        )
        self.actor_optimizer = Ranger(
            self.actor.parameters(),
            lr=cfg.actor_learning_rate,
            weight_decay=cfg.actor_weight_decay,
        )
        self.critic_optimizer = Ranger(
            self.critic.parameters(),
            lr=cfg.critic_learning_rate,
            weight_decay=cfg.critic_weight_decay,
        )

        t_max = cfg.max_episodes
        self.embedding_scheduler = CosineAnnealingLR(self.embedding_optimizer, T_max=t_max, eta_min=1e-5)
        self.actor_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=t_max, eta_min=1e-5)
        self.critic_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=t_max, eta_min=1e-5)

    # ================================================================== #
    # Reward shaping                                                      #
    # ================================================================== #
    def compute_shaped_reward(
        self,
        raw_reward: float,
        action: int,
        history: Sequence[int],
    ) -> float:
        """Augment the raw environment reward with diversity & coverage bonuses.

        Args:
            raw_reward: The reward delivered by the environment.
            action: The recommended item id.
            history: The current item history (excluding ``action``).

        Returns:
            The shaped reward.
        """
        alpha = self.config.reward_diversity_alpha
        beta = self.config.reward_coverage_beta
        diversity = 0.0 if action in history else 1.0
        coverage = 1.0
        return float(raw_reward) + alpha * diversity + beta * coverage



    # ================================================================== #
    # Target-network synchronisation                                      #
    # ================================================================== #
    @staticmethod
    def hard_update_target(source: nn.Module, target: nn.Module) -> None:
        """Copy ``source`` parameters into ``target`` in place."""
        with torch.no_grad():
            for p_src, p_tgt in zip(source.parameters(), target.parameters()):
                p_tgt.data.copy_(p_src.data)

    def soft_update_target(self, source: nn.Module, target: nn.Module) -> None:
        """Polyak-average ``source`` into ``target`` using ``τ``."""
        with torch.no_grad():
            for p_src, p_tgt in zip(source.parameters(), target.parameters()):
                p_tgt.data.mul_(1.0 - self.tau).add_(p_src.data, alpha=self.tau)



    # ================================================================== #
    # Action selection                                                    #
    # ================================================================== #
    def select_action(
        self,
        state: list,
        item_candidates: Optional[Sequence[int]] = None,
        top_k: int = 1,
        with_noise: bool = False,
    ):
        """Pick the top-``k`` items for the supplied state.

        Args:
            state: ``[group_id, *history_items]``.
            item_candidates: Optional candidate set; if ``None`` all items
                in the catalogue are scored.
            top_k: Number of items to recommend.
            with_noise: Add OU exploration noise to the action weight.

        Returns:
            A scalar item id when ``top_k == 1``, otherwise a numpy array.
        """
        with torch.no_grad():
            embedded_state = self.encode_state_batch([state])
            action_weight = torch.squeeze(self.actor(embedded_state))

            if with_noise:
                action_weight = action_weight + self.noise.sample().to(action_weight.device)

            if item_candidates is None:
                item_weights = self.embedding.item_embedding.weight.clone()
            else:
                item_candidates = np.asarray(item_candidates)
                idx_tensor = torch.as_tensor(item_candidates, dtype=torch.int, device=self.device)
                item_weights = self.embedding.item_embedding(idx_tensor)

            scores = torch.inner(action_weight, item_weights).detach().cpu().numpy()
            top_indices = np.argsort(scores)[::-1][:top_k].copy()

            chosen = top_indices if item_candidates is None else item_candidates[top_indices]
            chosen = np.squeeze(chosen)
            if top_k == 1:
                chosen = chosen.item()
            return chosen



    # ================================================================== #
    # State / action encoding                                             #
    # ================================================================== #
    def encode_state(self, state: list) -> torch.Tensor:
        """Encode a single state vector ``[group_id, *history]``."""
        group_id = state[0]
        members = torch.as_tensor(self.group_to_members[group_id], dtype=torch.int, device=self.device)
        history = torch.as_tensor(state[1:], dtype=torch.int, device=self.device)
        return self.embedding(members, history)

    def encode_state_batch(self, states: Iterable[list]) -> torch.Tensor:
        """Encode a batch of states by stacking individual encodings."""
        return torch.stack([self.encode_state(s) for s in states], dim=0)

    def encode_actions(self, actions: Sequence[int]) -> torch.Tensor:
        """Look up item embeddings for a batch of action ids."""
        action_t = torch.as_tensor(actions, dtype=torch.int, device=self.device)
        return self.embedding.item_embedding(action_t)

    def compute_embedded_actions(
        self,
        embedded_states: torch.Tensor,
        target: bool = False,
    ) -> torch.Tensor:
        """Differentiable item selection via Gumbel-softmax.

        Args:
            embedded_states: Tensor of encoded states.
            target: If ``True`` use the target actor; otherwise the live actor.
        """
        actor = self.actor_target if target else self.actor
        action_weights = actor(embedded_states)
        item_weights = self.embedding.item_embedding.weight.clone()
        scores = torch.inner(action_weights, item_weights)
        soft_one_hot = F.gumbel_softmax(scores, hard=True)
        return torch.inner(soft_one_hot, item_weights.t())



    # ================================================================== #
    # Optimisation                                                        #
    # ================================================================== #
    def update(self) -> Tuple[float, float]:
        """Run one DDPG optimisation step.

        Decomposed internally into :meth:`_update_critic` and
        :meth:`_update_actor`. The public method name is preserved for
        backward compatibility.

        Returns:
            ``(actor_loss, critic_loss)`` as Python floats.
        """
        batch, priority_indices, importance_weights = self.replay_buffer.sample(self.config.batch_size)
        importance_weights = importance_weights.to(self.device).unsqueeze(-1)  # [B, 1]

        states, actions, rewards, next_states = list(zip(*batch))

        critic_loss, embedded_states = self._update_critic(
            states=list(states),
            actions=list(actions),
            rewards=list(rewards),
            next_states=list(next_states),
            importance_weights=importance_weights,
            priority_indices=priority_indices,
        )
        actor_loss = self._update_actor(states=list(states))

        # Polyak averaging
        self.soft_update_target(self.actor, self.actor_target)
        self.soft_update_target(self.critic, self.critic_target)

        return actor_loss, critic_loss

    # ------------------------------------------------------------------ #
    def _update_critic(
        self,
        states: List[list],
        actions: List[int],
        rewards: List[float],
        next_states: List[list],
        importance_weights: torch.Tensor,
        priority_indices: np.ndarray,
    ) -> Tuple[float, torch.Tensor]:
        """One critic gradient step. Returns ``(loss, embedded_states)``."""
        self.critic_optimizer.zero_grad()

        embedded_states = self.encode_state_batch(states)
        embedded_actions = self.encode_actions(actions)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float, device=self.device).unsqueeze(-1)

        with torch.no_grad():
            embedded_next_states = self.encode_state_batch(next_states)
            embedded_next_actions = self.compute_embedded_actions(embedded_next_states, target=True)
            target_q = rewards_t + self.gamma * self.critic_target(embedded_next_states, embedded_next_actions)

        q_values = self.critic(embedded_states, embedded_actions)
        per_sample_loss = self.critic_criterion(q_values, target_q)
        loss = (importance_weights * per_sample_loss).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Refresh priorities
        td_errors = (q_values - target_q).abs().detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(priority_indices, td_errors)

        return float(loss.detach().cpu().item()), embedded_states.detach()

    # ------------------------------------------------------------------ #
    def _update_actor(self, states: List[list]) -> float:
        """One actor + integrator gradient step. Returns the actor loss."""
        self.actor_optimizer.zero_grad()
        self.embedding_optimizer.zero_grad()

        embedded_states = self.encode_state_batch(states)
        action_weights = self.actor(embedded_states)
        item_emb = self.embedding.item_embedding.weight
        scores = torch.inner(action_weights, item_emb)

        log_probs = torch.log_softmax(scores, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        embedded_actions = self.compute_embedded_actions(embedded_states)
        q_for_actor = self.critic(embedded_states, embedded_actions).mean()

        loss = -q_for_actor - self.config.entropy_coef * entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.embedding.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        self.embedding_optimizer.step()

        return float(loss.detach().cpu().item())

    # ================================================================== #
    # Validation diagnostics                                              #
    # ================================================================== #
    def evaluate_loss(
        self,
        df_eval: pd.DataFrame,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Compute actor and critic losses on held-out evaluation data.

        No network weights are updated. Public method name kept stable.

        Args:
            df_eval: Evaluation dataframe containing ``group``, ``history``
                and ``action`` columns.

        Returns:
            ``(actor_loss, critic_loss)`` or ``(None, None)`` if ``df_eval``
            is empty.
        """
        if df_eval is None or (isinstance(df_eval, pd.DataFrame) and df_eval.empty):
            return None, None

        sample_n = min(self.config.batch_size, len(df_eval))
        rows = df_eval.sample(n=sample_n, replace=False)

        states: List[list] = []
        actions: List[int] = []
        next_states: List[list] = []
        for _, row in rows.iterrows():
            group = row["group"]
            history = list(row["history"])
            action = row["action"]
            states.append([group] + history)
            actions.append(action)
            next_states.append([group] + history[1:] + [action])

        with torch.no_grad():
            embedded_states = self.encode_state_batch(states)
            embedded_actions = self.encode_actions(actions)
            rewards_t = torch.ones(len(states), 1, dtype=torch.float, device=self.device)
            embedded_next_states = self.encode_state_batch(next_states)

            q_values = self.critic(embedded_states, embedded_actions)
            embedded_next_actions = self.compute_embedded_actions(embedded_next_states, target=True)
            target_q = rewards_t + self.gamma * self.critic_target(embedded_next_states, embedded_next_actions)

            critic_loss = float(F.mse_loss(q_values, target_q).cpu().item())

            action_weights = self.actor(embedded_states)
            item_emb = self.embedding.item_embedding.weight.clone()
            scores = torch.inner(action_weights, item_emb)
            log_probs = torch.log_softmax(scores, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            embedded_actions_actor = self.compute_embedded_actions(embedded_states)
            q_for_actor = self.critic(embedded_states, embedded_actions_actor).mean()
            actor_loss = float((-q_for_actor - self.config.entropy_coef * entropy).cpu().item())

        return actor_loss, critic_loss






