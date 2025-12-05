import os
import csv
from collections import Counter
from typing import List, Dict, Any

import matplotlib.pyplot as plt


RESULTS_DIR = "results"
METRICS_CSV = os.path.join(RESULTS_DIR, "episode_metrics.csv")


def load_metrics(csv_path: str) -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    with open(csv_path, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields where appropriate
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


def plot_reward_vs_episode(metrics: List[Dict[str, Any]], output_path: str) -> None:
    episodes = [m["episode_index"] + 1 for m in metrics]
    rewards = [m["total_reward"] for m in metrics]

    ma_rewards = moving_average(rewards, window=20)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, alpha=0.3, label="Reward per episode")
    plt.plot(episodes, ma_rewards, linewidth=2, label="Moving average (20 eps)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Reward Over Time (DQN + Bandit)")
    plt.legend()
    plt.grid(True, alpha=0.2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved reward plot to: {output_path}")


def plot_bugs_vs_episode(metrics: List[Dict[str, Any]], output_path: str) -> None:
    episodes = [m["episode_index"] + 1 for m in metrics]
    bugs = [m["bugs_found"] for m in metrics]
    ma_bugs = moving_average(bugs, window=20)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, bugs, alpha=0.4, label="Bugs found per episode")
    plt.plot(episodes, ma_bugs, linewidth=2, label="Moving average (20 eps)")
    plt.xlabel("Episode")
    plt.ylabel("Bugs Found")
    plt.title("Bugs Found vs Episode")
    plt.legend()
    plt.grid(True, alpha=0.2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved bugs plot to: {output_path}")


def plot_strategy_usage(metrics: List[Dict[str, Any]], output_path: str) -> None:
    strategy_names = [m["strategy_name"] for m in metrics]
    counts = Counter(strategy_names)

    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.xlabel("Strategy")
    plt.ylabel("Episodes Used")
    plt.title("Strategy Usage Across Episodes")
    plt.grid(True, axis="y", alpha=0.2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved strategy usage plot to: {output_path}")


def main() -> None:
    if not os.path.exists(METRICS_CSV):
        print(f"[Error] Metrics CSV not found at: {METRICS_CSV}")
        print("Run `python -m src.main` first to generate episode_metrics.csv.")
        return

    metrics = load_metrics(METRICS_CSV)
    print(f"[Info] Loaded {len(metrics)} episodes from {METRICS_CSV}")

    reward_png = os.path.join(RESULTS_DIR, "reward_vs_episode.png")
    bugs_png = os.path.join(RESULTS_DIR, "bugs_found_vs_episode.png")
    strategy_png = os.path.join(RESULTS_DIR, "strategy_usage.png")

    plot_reward_vs_episode(metrics, reward_png)
    plot_bugs_vs_episode(metrics, bugs_png)
    plot_strategy_usage(metrics, strategy_png)

    print("\nAll plots generated in the 'results/' folder.")


if __name__ == "__main__":
    main()
