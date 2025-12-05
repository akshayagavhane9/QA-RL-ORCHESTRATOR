"""
Test Planner Agent (DQN-based) for QA-RL Orchestrator.

This agent will:
- Observe the CI environment state
- Choose which test to run next (action)
- Use a DQN model to approximate Q(s, a)
- Update its policy based on experience replay
"""

from typing import Dict, Any

from src.config import settings
from src.rl.dqn import DQNAgent


class TestPlannerAgent:
    """
    DQN-based agent that selects the next test to run in the CI pipeline.

    It interfaces with:
    - CISimulationEnvironment (for state and reward)
    - DQNAgent (for learning and action selection)
    """

    def __init__(self) -> None:
        self.num_tests = settings.NUM_TESTS

        # Our state encoder in DQNAgent currently produces a 12-dimensional vector:
        # [time_norm, tests_norm, bugs_norm] + 6 module flags + 3 size flags
        self.state_dim = 12
        self.action_dim = self.num_tests

        # Under-the-hood DQN model
        self.dqn_agent = DQNAgent(state_dim=self.state_dim, action_dim=self.action_dim)

    def select_action(self, state: Dict[str, Any]) -> int:
        """
        Given the current state, select the next test index to run.

        Args:
            state: dictionary representation of the CI state

        Returns:
            action: index of the chosen test (0 <= action < num_tests)
        """
        action = self.dqn_agent.select_action(state)

        # Safety check: clamp into valid range
        if action < 0 or action >= self.num_tests:
            action = action % self.num_tests

        return action

    def observe_transition(
        self,
        state: Dict[str, Any],
        action: int,
        reward: float,
        next_state: Dict[str, Any],
        done: bool,
    ) -> None:
        """
        Record a transition so the DQN can learn from it.

        Args:
            state: previous state
            action: action taken
            reward: reward received
            next_state: resulting state
            done: whether the episode finished after this step
        """
        self.dqn_agent.store_transition(state, action, reward, next_state, done)
        # We will implement full train_step logic in the DQN agent later (Checkpoint 4)
        self.dqn_agent.train_step()

    def end_episode(self) -> None:
        """
        Optional hook called at the end of each episode.

        For now, we simply update the target network periodically from outside.
        This method is a placeholder in case we want per-episode logic later.
        """
        # Could be used for episode-level logging or epsilon scheduling
        pass
