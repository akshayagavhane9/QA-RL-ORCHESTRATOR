"""
Controller Agent for QA-RL Orchestrator.

The ControllerAgent coordinates:
- Strategy selection (which high-level test strategy to use)
- Test planning (which individual tests to run, via the DQN planner)
- Interaction with the CI simulation environment

It does NOT implement RL directly.
Instead, it calls:
- StrategySelectorAgent (bandit)
- TestPlannerAgent (DQN-based)
"""

from typing import Optional, Dict, Any

from src.config.settings import TrainingConfig
from src.env.ci_env import CISimulationEnvironment
from src.agents.strategy_selector_agent import StrategySelectorAgent
from src.agents.test_planner_agent import TestPlannerAgent


class ControllerAgent:
    """
    High-level orchestrator for one or more CI episodes.

    Responsibilities:
    - Initialize the CI environment
    - For each episode:
        - Ask strategy selector which strategy to use
        - Coordinate test planner decisions (step-by-step)
        - Send actions to the environment
        - Collect episode metrics and return them for training/logging
    """

    def __init__(
        self,
        env: CISimulationEnvironment,
        strategy_agent: StrategySelectorAgent,
        planner_agent: TestPlannerAgent,
        config: Optional[TrainingConfig] = None,
    ) -> None:
        self.env = env
        self.strategy_agent = strategy_agent
        self.planner_agent = planner_agent
        self.config = config or TrainingConfig()

    def run_episode(self, episode_index: int) -> Dict[str, Any]:
        """
        Run a single CI episode.

        Flow:
        - Reset environment → get initial state
        - StrategySelectorAgent chooses a high-level strategy
        - Loop:
            - TestPlannerAgent selects an action (test index)
            - Environment executes the test and returns next_state, reward, done
            - TestPlannerAgent observes the transition for DQN learning
        - At the end, notify StrategySelectorAgent of the episode reward
        - Return episode-level metrics
        """
        # 1) Reset environment
        state = self.env.reset()

        # 2) Select strategy for this episode
        strategy_info = self.strategy_agent.select_strategy(episode_index=episode_index)
        strategy_index = strategy_info["strategy_index"]
        strategy_name = strategy_info["strategy_name"]

        # (Optional) attach strategy info to the state if we want later
        state["strategy_index"] = strategy_index
        state["strategy_name"] = strategy_name

        total_reward = 0.0
        steps = 0
        done = False

        # Track some metrics
        initial_changed_module = state.get("changed_module")
        initial_change_size = state.get("change_size")

        while not done and steps < self.config.max_episode_steps:
            action = self.planner_agent.select_action(state)

            next_state, reward, done, info = self.env.step(action)

            # Let the planner observe the transition (for DQN training)
            self.planner_agent.observe_transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            total_reward += reward
            steps += 1

            # Move to next state
            state = next_state

        # Episode finished – optional hook for planner
        self.planner_agent.end_episode()

        # Tell the strategy selector how well this strategy did
        self.strategy_agent.update_with_episode_result(
            strategy_index=strategy_index,
            episode_reward=total_reward,
        )

        # Build episode metrics
        episode_metrics: Dict[str, Any] = {
            "episode_index": episode_index,
            "total_reward": total_reward,
            "total_time": self.env.time_used,
            "tests_run": len(self.env.tests_run),
            "bugs_found": self.env.bugs_found,
            "strategy_index": strategy_index,
            "strategy_name": strategy_name,
            "changed_module": initial_changed_module,
            "change_size": initial_change_size,
        }
        return episode_metrics
