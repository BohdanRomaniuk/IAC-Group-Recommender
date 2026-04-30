"""
Critic network for the DDPG agent.
"""
from typing import Tuple

import torch
import torch.nn as nn


class Critic(nn.Module):
    """
    Critic Network with LayerNorm and Dropout to stabilise training.
    """

    def __init__(self, embedded_state_size: int, embedded_action_size: int, hidden_sizes: Tuple[int, ...]):
        """
        Initialize Critic

        :param embedded_state_size: embedded state size
        :param embedded_action_size: embedded action size
        :param hidden_sizes: hidden sizes
        """
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(embedded_state_size + embedded_action_size, hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_sizes[0], hidden_sizes[0]),  # extra hidden layer
            nn.LayerNorm(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LayerNorm(hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
            # No Tanh: unbounded Q-values are required for correct Bellman updates
        )

    def forward(self, embedded_state: torch.Tensor, embedded_action: torch.Tensor) -> torch.Tensor:
        """
        Forward

        :param embedded_state: embedded state
        :param embedded_action: embedded action
        :return: Q value
        """
        return self.net(torch.cat([embedded_state, embedded_action], dim=-1))

