"""
Entry point for the QA-RL Orchestrator project.

This script is used to:
- Initialize the CI simulation environment
- Create agents (controller, strategy selector, test planner)
- Run training loops and/or evaluation runs
- Eventually drive the demo for the assignment
"""

from typing import Optional, List, Dict, Any
import os
import csv

from src.config.settings import TrainingConfig, NUM_EPISODES
from src.env.ci_env import CISimulationEnvironment
from src.agents.controller_agent import ControllerAgent
from src.agents.strategy_selector_agent import StrategySelectorAgent
from src.agents.test_planner_agent import TestPlannerAgent


def _save_metrics_to_csv(metrics: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save episode-level metrics to a CSV file for later analysis / plotting.

    Each dict in `metrics` is expected to have the same keys.
    """
    if not metrics:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = list(metrics[0].keys())
    with open(output_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)

    print(f"\n[Metrics] Saved episode metrics to: {output_path}")


def run_training(config: Optional[TrainingConfig] = None) -> List[Dict[str, Any]]:
    """
    Main training entry point.

    Steps:
    - Initialize environment and agents
    - For each episode:
        - run a CI simulation via ControllerAgent
        - collect episode-level metrics
    - Save metrics to CSV
    - Return the list of metrics for later analysis/plotting
    """
    if config is None:
        config = TrainingConfig()
        # Ensure num_episodes is aligned with global constant
        config.num_episodes = NUM_EPISODES

    print("=== QA-RL Orchestrator ===")
    print(f"Starting training for {config.num_episodes} episodes.\n")

    # 1) Initialize environment and agents
    env = CISimulationEnvironment()
    strategy_agent = StrategySelectorAgent()
    planner_agent = TestPlannerAgent()
    controller = ControllerAgent(env, strategy_agent, planner_agent, config=config)

    all_metrics: List[Dict[str, Any]] = []

    # 2) Training loop over episodes
    for episode_idx in range(config.num_episodes):
        episode_metrics = controller.run_episode(episode_index=episode_idx)
        all_metrics.append(episode_metrics)

        # Simple progress logging
        if (episode_idx + 1) % 50 == 0 or episode_idx == 0:
            print(
                f"[Episode {episode_idx + 1}/{config.num_episodes}] "
                f"Reward={episode_metrics['total_reward']:.2f}, "
                f"Time={episode_metrics['total_time']:.1f}s, "
                f"Tests={episode_metrics['tests_run']}, "
                f"BugsFound={episode_metrics['bugs_found']}, "
                f"Strategy={episode_metrics['strategy_name']}"
            )

    print("\nTraining finished.")

    # 3) Save metrics to CSV for analysis / plotting
    output_csv = os.path.join("results", "episode_metrics.csv")
    _save_metrics_to_csv(all_metrics, output_csv)

    return all_metrics


def run_demo() -> None:
    """
    Run a short demo episode or evaluation run.

    This will be useful for your 10-minute video later.
    For now, it's just a placeholder.
    """
    print("Running demo (placeholder).")
    # Later:
    # - load trained models
    # - run a few CI episodes
    # - print out which tests are selected and why


if __name__ == "__main__":
    # For now we just run training when executed as a script.
    # Later, you can add argparse to choose between 'train' and 'demo'.
    run_training()
