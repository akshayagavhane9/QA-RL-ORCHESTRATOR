"""
Global configuration for the QA-RL Orchestrator project.

These settings control:
- Test suite simulation properties
- CI episode configuration
- Reward function parameters
- RL hyperparameters (DQN + UCB Bandit)
"""

from dataclasses import dataclass


# ============================================================
#  TEST SUITE & CI ENVIRONMENT SETTINGS
# ============================================================

# Total number of tests in our simulated suite
NUM_TESTS = 15

# Maximum number of actions (test selections) per CI episode
MAX_EPISODE_STEPS = 20

# Maximum allowed CI time (simulated)
TIME_BUDGET_SECONDS = 300.0



# ============================================================
#  DQN REWARD FUNCTION PARAMETERS (Test Planner Agent)
# ============================================================

# Reward when a test catches a REAL bug (not a flaky failure)
REWARD_BUG_FOUND = 10.0

# Penalty based on how long a test takes (duration cost)
PENALTY_PER_SECOND = 0.05

# Small penalty per test executed (discourages running all tests blindly)
PENALTY_PER_TEST = 0.1

# Penalty if the episode ends but a real bug was NOT detected
PENALTY_MISSED_BUG = 20.0

# Penalty if CI time budget is exceeded
PENALTY_OVERTIME = 10.0



# ============================================================
#  DQN HYPERPARAMETERS
# ============================================================

DQN_GAMMA = 0.99               # Discount factor
DQN_LR = 1e-3                  # Learning rate
DQN_BATCH_SIZE = 64           
DQN_REPLAY_BUFFER_SIZE = 50_000

# Update target network every X steps
DQN_TARGET_UPDATE_FREQ = 1000

# Epsilon-greedy exploration schedule
DQN_EPSILON_START = 1.0
DQN_EPSILON_END = 0.1
DQN_EPSILON_DECAY_STEPS = 10_000



# ============================================================
#  STRATEGY BANDIT (UCB) SETTINGS
# ============================================================

# How many high-level strategies we support
NUM_STRATEGIES = 5  

# Exploration constant for UCB formula
UCB_CONFIDENCE_C = 2.0  

# Names of the strategies ("arms" in bandit language)
STRATEGY_NAMES = [
    "smoke_first",
    "api_first",
    "ui_first",
    "high_risk_first",
    "baseline_fixed_order",
]



# ============================================================
#  EXPERIMENT SETTINGS
# ============================================================

# How many CI episodes (commits) to simulate during training
NUM_EPISODES = 500



# ============================================================
#  OPTIONAL STRUCTURED TRAINING CONFIG (for cleaner passing around)
# ============================================================

@dataclass
class TrainingConfig:
    """
    Structured configuration for training.
    You can pass this around instead of using the global constants directly.
    """
    num_episodes: int = NUM_EPISODES
    max_episode_steps: int = MAX_EPISODE_STEPS
    time_budget_seconds: float = TIME_BUDGET_SECONDS
    dqn_gamma: float = DQN_GAMMA
    dqn_lr: float = DQN_LR
    dqn_batch_size: int = DQN_BATCH_SIZE
