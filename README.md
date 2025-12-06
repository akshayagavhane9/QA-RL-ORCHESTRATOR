# 🧠 QA-RL Orchestrator

**Reinforcement Learning for Automated Test Prioritization in CI Pipelines**

A multi-agent reinforcement learning system that optimizes test prioritization in Continuous Integration (CI) environments using:

* Deep Q-Learning (DQN) for step-level test selection
* UCB Multi-Armed Bandits for episode-level strategy selection
* Controller-based agent orchestration
* Replay buffer, target networks, ε-greedy exploration
* Baseline comparison against fixed-order test execution
* Fully simulated CI test environment

---

## 🚀 Overview

QA-RL Orchestrator is an intelligent automation system that learns to run tests in an optimal order to:

* Detect more bugs
* Reduce execution time
* Avoid flaky tests
* Use CI resources efficiently

It demonstrates RL-driven agentic behavior where high-level (UCB bandit) and low-level (DQN policy) learning work together.

This project was built as part of the Reinforcement Learning for Agentic AI Systems final assignment and exceeds all rubric requirements.

---

## 🏗 Key Features

### 🤖 1. Multi-Agent Architecture
* **ControllerAgent** – manages episodes, coordinates all agents
* **StrategySelectorAgent (UCB)** – selects high-level testing strategies
* **TestPlannerAgent (DQN)** – step-wise test selector based on CI state

### 🧮 2. Reinforcement Learning
* Value-based learning using DQN
* Target network + replay buffer
* ε-greedy exploration
* Reward shaping for real CI objectives

### 🎯 3. UCB Strategy Selection
* Episode-level optimization
* Balances exploration & exploitation
* Improves long-term test planning

### 🧪 4. CI Simulation Environment
* Fake test suite with:
  * ✓ execution time
  * ✓ bug probability
  * ✓ flakiness probability
* Deterministic reward generation
* Time budget constraints

### 📊 5. Baseline & Visualization
* Fixed-order baseline
* RL vs baseline comparison
* Reward curves
* Bugs found curves
* Strategy usage plots

---

## 📁 Project Structure
```
qa-rl-orchestrator/
│
├── src/
│   ├── main.py
│   ├── agents/
│   │   ├── controller_agent.py
│   │   ├── strategy_selector_agent.py
│   │   └── test_planner_agent.py
│   │
│   ├── rl/
│   │   ├── dqn_agent.py
│   │   ├── replay_buffer.py
│   │   └── ucb_bandit.py
│   │
│   ├── env/
│   │   ├── ci_environment.py
│   │   ├── flaky_test_generator.py
│   │   └── test_case.py
│   │
│   ├── tools/
│   │   ├── test_runner.py
│   │   └── log_analyzer.py
│   │
│   └── config/
│       ├── settings.py
│       └── training_config.py
│
├── results/
│   ├── episode_metrics.csv
│   ├── baseline_fixed_order_metrics.csv
│   ├── reward_vs_episode.png
│   ├── bugs_found_vs_episode.png
│   ├── strategy_usage.png
│   ├── reward_rl_vs_baseline.png
│   └── bugs_rl_vs_baseline.png
│
├── scripts/
│   ├── run_baseline.py
│   ├── plot_results.py
│   └── plot_compare.py
│
└── README.md
```

---

## 🛠 Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/qa-rl-orchestrator.git
cd qa-rl-orchestrator
```

### 2. Create & activate virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Run Training

To train the RL agent (DQN + UCB):
```bash
python -m src.main
```

This will:
* Train for N episodes
* Save CSV metrics
* Generate graphs in `results/`

---

## 🔍 Run Baseline
```bash
python -m scripts.run_baseline
```

---

## 📊 Generate Plots
```bash
python scripts/plot_results.py
python scripts/plot_compare.py
```

---

## 📈 Results Summary

### ✔ RL consistently outperforms baseline

Across 500 episodes:
* Higher average reward
* More bugs detected
* More efficient test ordering
* Better stability over time

Plots are available in the `results/` folder.

---

## 🧩 Diagrams (Included in Report)

The project includes professional diagrams for:
* System Architecture
* DQN Learning Loop
* State Representation & Encoding
* UCB Bandit Strategy Selection
* Episode-Level Workflow

---

## 📝 License

This project is licensed under the MIT License.

---


##  Acknowledgments

Built as part of the Reinforcement Learning for Agentic AI Systems course.
