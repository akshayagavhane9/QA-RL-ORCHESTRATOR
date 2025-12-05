"""
DQN implementation for the Test Planner Agent in QA-RL Orchestrator.

This module defines:
- A simple neural network for Q-value approximation
- A DQNAgent class that:
    - selects actions (epsilon-greedy)
    - stores transitions in a replay buffer
    - performs training steps
"""

from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import settings
from src.rl.replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    """
    Simple feedforward network to approximate Q(s, a).
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """
    DQN agent that learns Q(s, a) for the test planner.

    The test planner will:
    - call select_action() to choose a test index
    - call store_transition() after each environment step
    - call train_step() periodically to update the Q-network
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Main and target networks
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=settings.DQN_LR
        )

        self.gamma = settings.DQN_GAMMA

        self.replay_buffer = ReplayBuffer(settings.DQN_REPLAY_BUFFER_SIZE)
        self.batch_size = settings.DQN_BATCH_SIZE
        self.target_update_freq = settings.DQN_TARGET_UPDATE_FREQ

        # Epsilon-greedy exploration
        self.epsilon = settings.DQN_EPSILON_START
        self.epsilon_start = settings.DQN_EPSILON_START
        self.epsilon_end = settings.DQN_EPSILON_END
        self.epsilon_decay_steps = settings.DQN_EPSILON_DECAY_STEPS
        self.total_steps = 0

        self.loss_fn = nn.MSELoss()

        # Count how many gradient update steps we’ve done
        self.training_steps = 0

    # ---------------------------------------------------------------------
    # State encoding
    # ---------------------------------------------------------------------

    def _state_to_tensor(self, state: Dict[str, Any]) -> torch.Tensor:
        """
        Convert a state dict into a fixed-length 1D tensor.

        State fields expected:
        - time_used: float
        - tests_run_count: int
        - bugs_found: int
        - changed_module: str
        - change_size: str
        """
        # Numeric features (normalized)
        time_used = float(state.get("time_used", 0.0))
        tests_run_count = float(state.get("tests_run_count", 0))
        bugs_found = float(state.get("bugs_found", 0))

        time_norm = min(time_used / settings.TIME_BUDGET_SECONDS, 1.0)
        tests_norm = min(tests_run_count / settings.MAX_EPISODE_STEPS, 1.0)
        bugs_norm = min(bugs_found / 3.0, 1.0)  # assume typically <= 3 bugs caught

        # One-hot encoding for changed_module
        modules = ["auth", "checkout", "search", "payments", "profile", "notifications"]
        changed_module = state.get("changed_module", None)
        module_one_hot = [1.0 if changed_module == m else 0.0 for m in modules]

        # One-hot encoding for change_size
        size_options = ["small", "medium", "large"]
        change_size = state.get("change_size", None)
        size_one_hot = [1.0 if change_size == s else 0.0 for s in size_options]

        # Final feature vector of length 12:
        # [time_norm, tests_norm, bugs_norm] + 6 module flags + 3 size flags
        features = [time_norm, tests_norm, bugs_norm] + module_one_hot + size_one_hot

        arr = np.array(features, dtype=np.float32)  # shape: (12,)
        return torch.from_numpy(arr).unsqueeze(0)  # shape: (1, state_dim)

    # ---------------------------------------------------------------------
    # Action selection (epsilon-greedy)
    # ---------------------------------------------------------------------

    def select_action(self, state: Dict[str, Any]) -> int:
        """
        Epsilon-greedy action selection.
        """
        self.total_steps += 1

        # Update epsilon linearly from start -> end
        fraction = min(1.0, self.total_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

        # Exploration
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        # Exploitation
        state_tensor = self._state_to_tensor(state)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        action = int(torch.argmax(q_values, dim=1).item())
        return action

    # ---------------------------------------------------------------------
    # Replay buffer interaction
    # ---------------------------------------------------------------------

    def store_transition(
        self,
        state: Dict[str, Any],
        action: int,
        reward: float,
        next_state: Dict[str, Any],
        done: bool,
    ) -> None:
        """
        Store a transition in the replay buffer.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    # ---------------------------------------------------------------------
    # Training logic
    # ---------------------------------------------------------------------

    def train_step(self) -> None:
        """
        Perform a single DQN training step if enough data is available.

        This implements the standard DQN update:
        - Sample a minibatch from replay buffer
        - Compute Q(s, a) for taken actions
        - Compute target = r + gamma * max_a' Q_target(s', a')  (if not done)
        - Minimize MSE between Q and target
        - Periodically update target network
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # ---- Convert batch to tensors ----
        # Stack state tensors
        state_tensors: List[torch.Tensor] = [
            self._state_to_tensor(s) for s in states
        ]  # list of (1, state_dim)
        states_tensor = torch.cat(state_tensors, dim=0)  # (B, state_dim)

        next_state_tensors: List[torch.Tensor] = [
            self._state_to_tensor(ns) for ns in next_states
        ]
        next_states_tensor = torch.cat(next_state_tensors, dim=0)  # (B, state_dim)

        actions_tensor = torch.tensor(actions, dtype=torch.long)  # (B,)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)  # (B,)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)  # (B,)

        # ---- Compute current Q-values for chosen actions ----
        q_values = self.q_network(states_tensor)  # (B, action_dim)
        # gather the Q-value corresponding to each action in the batch
        q_values_chosen = q_values.gather(
            1, actions_tensor.unsqueeze(1)
        ).squeeze(1)  # (B,)

        # ---- Compute target Q-values using target network ----
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor)  # (B, action_dim)
            max_next_q_values, _ = torch.max(next_q_values, dim=1)  # (B,)

            targets = rewards_tensor + self.gamma * (1.0 - dones_tensor) * max_next_q_values

        # ---- Compute loss and backprop ----
        loss = self.loss_fn(q_values_chosen, targets)

        self.optimizer.zero_grad()
        loss.backward()

        # Optional: gradient clipping to improve stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.step()

        self.training_steps += 1

        # ---- Periodically update target network ----
        if self.training_steps % self.target_update_freq == 0:
            self.update_target_network()

    def update_target_network(self) -> None:
        """
        Copy parameters from the main Q-network to the target network.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
