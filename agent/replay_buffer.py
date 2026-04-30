"""
Prioritized Experience Replay buffer.
"""
import numpy as np
import torch


class PrioritizedReplayBuffer(object):
    """
    Prioritized Experience Replay Buffer (proportional variant).

    Transitions with higher TD-error are sampled more often.
    Importance-sampling (IS) weights correct the resulting bias.

    References:
        Schaul et al., "Prioritized Experience Replay", ICLR 2016.
    """

    def __init__(self, buffer_size: int, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_frames: int = 100_000):
        """
        Initialize PrioritizedReplayBuffer

        :param buffer_size: maximum number of stored transitions
        :param alpha: priority exponent (0 = uniform, 1 = full prioritisation)
        :param beta_start: initial IS-weight exponent (annealed to 1 over training)
        :param beta_frames: number of push() calls over which beta reaches 1.0
        """
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames

        self.buffer: list = []
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.pos = 0  # circular write pointer

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, experience: tuple) -> None:
        """
        Push one experience into the buffer with maximum current priority.

        :param experience: (state, action, reward, next_state)
        """
        max_priority = float(self.priorities[:len(self.buffer)].max()) if self.buffer else 1.0

        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.buffer_size

    def sample(self, batch_size: int):
        """
        Sample a batch proportional to stored priorities.

        :param batch_size: number of transitions to sample
        :return: (batch list, indices, IS-weight tensor)
        """
        n = len(self.buffer)
        priorities = self.priorities[:n]

        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)

        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        batch = [self.buffer[i] for i in indices]
        is_weights = torch.FloatTensor(weights)
        return batch, indices, is_weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray, epsilon: float = 1e-6) -> None:
        """
        Update priorities for a previously sampled batch.

        :param indices: indices returned by sample()
        :param td_errors: absolute TD errors for the sampled transitions
        :param epsilon: small constant added to prevent zero priorities
        """
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = float(abs(err)) + epsilon

