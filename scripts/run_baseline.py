import os
import csv
import random
from typing import List, Dict, Any

from src.config.settings import TrainingConfig, NUM_EPISODES
from src.env.ci_env import CISimulationEnvironment


RESULTS_DIR = "results"
BASELINE_CSV = os.path.join(RESULTS_DIR, "baseline_fixed_order_metrics.csv")


def run_baseline_fixed_order(num_episodes: int) -> List[Dict[str, Any]]:
    """
    Run a simple baseline strategy:
    - No learning
    - For each episode:
        - Always run tests in fixed index order (0, 1, 2, ..., N-1, 0, 1, ...)
    """
    config = TrainingConfig()
    config.num_episodes = num_episodes

    all_metrics: List[Dict[str, Any]] = []

    print(f"=== Baseline: Fixed Test Order ===")
    print(f"Running {num_episodes} baseline episodes.\n")

    for episode_idx in range(num_episodes):
        env = CISimulationEnvironment()
        state = env.reset()

        done = False
        steps = 0
        total_reward = 0.0

        initial_changed_module = state.get("changed_module")
        initial_change_size = state.get("change_size")

        while not done and steps < config.max_episode_steps:
            # Fixed-order baseline: cycle through tests
            action = steps % env.num_tests

            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            state = next_state

        episode_metrics: Dict[str, Any] = {
            "episode_index": episode_idx,
            "total_reward": total_reward,
            "total_time": env.time_used,
            "tests_run": len(env.tests_run),
            "bugs_found": env.bugs_found,
            "strategy_index": -1,
            "strategy_name": "baseline_fixed_order",
            "changed_module": initial_changed_module,
            "change_size": initial_change_size,
        }
        all_metrics.append(episode_metrics)

        if (episode_idx + 1) % 50 == 0 or episode_idx == 0:
            print(
                f"[Baseline Episode {episode_idx + 1}/{num_episodes}] "
                f"Reward={total_reward:.2f}, "
                f"Time={env.time_used:.1f}s, "
                f"Tests={len(env.tests_run)}, "
                f"BugsFound={env.bugs_found}"
            )

    print("\nBaseline run finished.")
    return all_metrics


def save_baseline_metrics_to_csv(metrics: List[Dict[str, Any]], output_path: str) -> None:
    if not metrics:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = list(metrics[0].keys())
    with open(output_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)

    print(f"[Baseline Metrics] Saved to: {output_path}")


def main() -> None:
    num_episodes = NUM_EPISODES  # keep same as RL training for fair comparison
    metrics = run_baseline_fixed_order(num_episodes=num_episodes)
    save_baseline_metrics_to_csv(metrics, BASELINE_CSV)


if __name__ == "__main__":
    main()
