"""
Replay buffer for DQN in QA-RL Orchestrator.

Stores transitions of the form:
    (state, action, reward, next_state, done)
so that we can sample random minibatches for training.
"""

from typing import List, Tuple, Dict, Any
import random


class ReplayBuffer:
    """
    Simple replay buffer implementation.

    This will be used by the DQNAgent to store experience and
    sample batches for learning.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: List[Tuple[Dict[str, Any], int, float, Dict[str, Any], bool]] = []
        self.position = 0

    def push(
        self,
        state: Dict[str, Any],
        action: int,
        reward: float,
        next_state: Dict[str, Any],
        done: bool,
    ) -> None:
        """
        Save a transition in the buffer.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int
    ) -> List[Tuple[Dict[str, Any], int, float, Dict[str, Any], bool]]:
        """
        Sample a random minibatch of transitions.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
