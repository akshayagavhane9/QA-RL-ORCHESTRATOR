import os
import csv
from typing import List, Dict, Any

import matplotlib.pyplot as plt


RESULTS_DIR = "results"
RL_CSV = os.path.join(RESULTS_DIR, "episode_metrics.csv")
BASELINE_CSV = os.path.join(RESULTS_DIR, "baseline_fixed_order_metrics.csv")


def load_metrics(csv_path: str) -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    with open(csv_path, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["episode_index"] = int(row["episode_index"])
            row["total_reward"] = float(row["total_reward"])
            row["total_time"] = float(row["total_time"])
            row["tests_run"] = int(row["tests_run"])
            row["bugs_found"] = int(row["bugs_found"])
            metrics.append(row)
    return metrics


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values
    ma = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_slice = values[start : i + 1]
        ma.append(sum(window_slice) / len(window_slice))
    return ma


def plot_reward_comparison(
    rl_metrics: List[Dict[str, Any]],
    baseline_metrics: List[Dict[str, Any]],
    output_path: str,
) -> None:
    rl_episodes = [m["episode_index"] + 1 for m in rl_metrics]
    rl_rewards = [m["total_reward"] for m in rl_metrics]
    rl_ma = moving_average(rl_rewards, window=20)

    base_episodes = [m["episode_index"] + 1 for m in baseline_metrics]
    base_rewards = [m["total_reward"] for m in baseline_metrics]
    base_ma = moving_average(base_rewards, window=20)

    plt.figure(figsize=(10, 5))
    plt.plot(rl_episodes, rl_ma, linewidth=2, label="RL (DQN + Bandit)")
    plt.plot(base_episodes, base_ma, linewidth=2, linestyle="--", label="Baseline (Fixed Order)")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward (Moving Avg, 20 eps)")
    plt.title("RL vs Baseline: Episode Reward")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Compare Plot] Saved reward comparison to: {output_path}")


def plot_bugs_comparison(
    rl_metrics: List[Dict[str, Any]],
    baseline_metrics: List[Dict[str, Any]],
    output_path: str,
) -> None:
    rl_episodes = [m["episode_index"] + 1 for m in rl_metrics]
    rl_bugs = [m["bugs_found"] for m in rl_metrics]
    rl_ma = moving_average(rl_bugs, window=20)

    base_episodes = [m["episode_index"] + 1 for m in baseline_metrics]
    base_bugs = [m["bugs_found"] for m in baseline_metrics]
    base_ma = moving_average(base_bugs, window=20)

    plt.figure(figsize=(10, 5))
    plt.plot(rl_episodes, rl_ma, linewidth=2, label="RL (DQN + Bandit)")
    plt.plot(base_episodes, base_ma, linewidth=2, linestyle="--", label="Baseline (Fixed Order)")

    plt.xlabel("Episode")
    plt.ylabel("Bugs Found (Moving Avg, 20 eps)")
    plt.title("RL vs Baseline: Bugs Found per Episode")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Compare Plot] Saved bugs comparison to: {output_path}")


def print_summary_stats(
    rl_metrics: List[Dict[str, Any]],
    baseline_metrics: List[Dict[str, Any]],
) -> None:
    rl_avg_reward = sum(m["total_reward"] for m in rl_metrics) / len(rl_metrics)
    base_avg_reward = sum(m["total_reward"] for m in baseline_metrics) / len(baseline_metrics)

    rl_avg_bugs = sum(m["bugs_found"] for m in rl_metrics) / len(rl_metrics)
    base_avg_bugs = sum(m["bugs_found"] for m in baseline_metrics) / len(baseline_metrics)

    print("\n=== Summary: RL vs Baseline ===")
    print(f"RL     - Avg Reward: {rl_avg_reward:.2f}, Avg Bugs Found: {rl_avg_bugs:.2f}")
    print(f"Baseline - Avg Reward: {base_avg_reward:.2f}, Avg Bugs Found: {base_avg_bugs:.2f}")


def main() -> None:
    if not os.path.exists(RL_CSV):
        print(f"[Error] RL metrics CSV not found at: {RL_CSV}")
        return
    if not os.path.exists(BASELINE_CSV):
        print(f"[Error] Baseline metrics CSV not found at: {BASELINE_CSV}")
        return

    rl_metrics = load_metrics(RL_CSV)
    baseline_metrics = load_metrics(BASELINE_CSV)
    print(f"[Info] Loaded {len(rl_metrics)} RL episodes and {len(baseline_metrics)} baseline episodes.")

    reward_png = os.path.join(RESULTS_DIR, "reward_rl_vs_baseline.png")
    bugs_png = os.path.join(RESULTS_DIR, "bugs_rl_vs_baseline.png")

    plot_reward_comparison(rl_metrics, baseline_metrics, reward_png)
    plot_bugs_comparison(rl_metrics, baseline_metrics, bugs_png)
    print_summary_stats(rl_metrics, baseline_metrics)

    print("\nAll comparison plots generated in the 'results/' folder.")


if __name__ == "__main__":
    main()
