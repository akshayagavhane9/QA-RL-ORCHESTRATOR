"""
Bandit algorithms for strategy selection in QA-RL Orchestrator.

We will use a simple UCB (Upper Confidence Bound) bandit
to select among high-level test strategies.
"""

from typing import List
import math
import random


class UCBBandit:
    """
    Upper Confidence Bound (UCB1) bandit implementation.

    Each "arm" corresponds to a high-level testing strategy.
    """

    def __init__(self, num_arms: int, c: float = 2.0) -> None:
        self.num_arms = num_arms
        self.c = c  # exploration coefficient

        self.counts: List[int] = [0 for _ in range(num_arms)]
        self.total_rewards: List[float] = [0.0 for _ in range(num_arms)]
        self.total_pulls: int = 0

    def select_arm(self) -> int:
        """
        Select which arm (strategy) to use next.

        Uses the UCB1 formula:
            UCB_i = mean_reward_i + c * sqrt(2 * ln(total_pulls) / pulls_i)
        """
        # First, play each arm at least once
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm

        self.total_pulls += 1

        ucb_values = []
        for arm in range(self.num_arms):
            mean_reward = self.total_rewards[arm] / self.counts[arm]
            bonus = self.c * math.sqrt(
                (2.0 * math.log(self.total_pulls)) / self.counts[arm]
            )
            ucb_values.append(mean_reward + bonus)

        # Choose arm with highest UCB value
        max_ucb = max(ucb_values)
        candidates = [i for i, val in enumerate(ucb_values) if val == max_ucb]
        return random.choice(candidates)

    def update(self, arm: int, reward: float) -> None:
        """
        Update statistics for the chosen arm based on observed reward.
        """
        self.counts[arm] += 1
        self.total_rewards[arm] += reward
        self.total_pulls += 1
