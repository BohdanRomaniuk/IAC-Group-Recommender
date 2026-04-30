"""
Actor network for the DDPG agent.
"""
from typing import Tuple

import torch.nn as nn


class Actor(nn.Module):
    """
    Actor Network with LayerNorm and Dropout for stability.
    """

    def __init__(self, embedded_state_size: int, action_weight_size: int, hidden_sizes: Tuple[int, ...]):
        """
        Initialize Actor

        :param embedded_state_size: embedded state size
        :param action_weight_size: embedded action size
        :param hidden_sizes: hidden sizes
        """
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(embedded_state_size, hidden_sizes[0]),
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
            nn.Dropout(p=0.1),
            nn.Linear(hidden_sizes[1], action_weight_size),
            nn.Tanh(),  # Bound action weights to [-1, 1]
        )

    def forward(self, embedded_state):
        """
        Forward

        :param embedded_state: embedded state
        :return: action weight
        """
        return self.net(embedded_state)

