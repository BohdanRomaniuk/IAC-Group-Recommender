"""
General-purpose helpers (currently: simple uniform replay memory).
"""
import random
from collections import deque


class ReplayMemory(object):
    """
    Simple uniform Replay Memory.
    """

    def __init__(self, buffer_size: int):
        """
        Initialize ReplayMemory

        :param buffer_size: maximum size of the buffer
        """
        self.buffer_size = buffer_size
        self.buffer: deque = deque(maxlen=buffer_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, experience: tuple) -> None:
        """
        Push one experience into the buffer.

        :param experience: (state, action, reward, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list:
        """
        Sample one batch from the buffer uniformly.

        :param batch_size: number of experiences in the batch
        :return: batch
        """
        return random.sample(self.buffer, batch_size)

