"""
Strategy Selector Agent (Bandit-based) for QA-RL Orchestrator.

This agent chooses a high-level test execution strategy for each CI episode,
such as:
- smoke_first
- api_first
- ui_first
- high_risk_first
- baseline_fixed_order

Internally, it uses a multi-armed bandit algorithm (UCB).
"""

from typing import Dict, Any

from src.config import settings
from src.rl.bandits import UCBBandit


class StrategySelectorAgent:
    """
    Bandit-based strategy selector.

    Responsibilities:
    - Maintain statistics for each strategy (via UCBBandit)
    - Select a strategy for each new episode
    - Update its estimates based on the episode-level reward
    """

    def __init__(self) -> None:
        self.strategy_names = settings.STRATEGY_NAMES
        self.num_strategies = len(self.strategy_names)

        # Under-the-hood UCB bandit
        self.bandit = UCBBandit(
            num_arms=self.num_strategies,
            c=settings.UCB_CONFIDENCE_C,
        )

    def select_strategy(self, episode_index: int) -> Dict[str, Any]:
        """
        Choose which strategy to use for the given episode.

        Args:
            episode_index: index of the current CI episode (for logging / exploration control)

        Returns:
            A dictionary containing:
            - strategy_index: int
            - strategy_name: str
        """
        strategy_index = self.bandit.select_arm()
        strategy_name = self.strategy_names[strategy_index]

        return {
            "strategy_index": strategy_index,
            "strategy_name": strategy_name,
        }

    def update_with_episode_result(
        self,
        strategy_index: int,
        episode_reward: float,
    ) -> None:
        """
        Update bandit statistics based on the observed reward for the chosen strategy.

        Args:
            strategy_index: which strategy was used
            episode_reward: total reward achieved in that episode
        """
        # Forward the update to UCB bandit
        self.bandit.update(strategy_index, episode_reward)
