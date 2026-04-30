"""
Ornstein-Uhlenbeck exploration noise.
"""
import torch

from config import Config


class OUNoise(object):
    """
    Ornstein-Uhlenbeck Noise (correlated noise for continuous action exploration).
    """

    def __init__(self, config: Config):
        """
        Initialize OUNoise

        :param config: configurations
        """
        self.embedded_action_size = config.embedded_action_size
        self.ou_mu = config.ou_mu
        self.ou_theta = config.ou_theta
        self.ou_sigma = config.ou_sigma
        self.ou_epsilon = config.ou_epsilon
        self.ou_state: torch.Tensor = torch.zeros(self.embedded_action_size)
        self.reset()

    def reset(self) -> None:
        """Reset the OU process state."""
        self.ou_state = torch.ones(self.embedded_action_size) * self.ou_mu

    def evolve_state(self) -> None:
        """Evolve the OU process state."""
        self.ou_state += self.ou_theta * (self.ou_mu - self.ou_state) \
            + self.ou_sigma * torch.randn(self.embedded_action_size)

    def get_ou_noise(self) -> torch.Tensor:
        """
        Get the OU noise for one action.

        :return: OU noise
        """
        self.evolve_state()
        return self.ou_state.clone()

