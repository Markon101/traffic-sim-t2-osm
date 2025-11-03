from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Tuple

import numpy as np
import torch
from torch import nn


class ReplayBuffer:
    """Experience replay storage for DQN."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(
            (
                state.astype(np.float32, copy=False),
                int(action),
                float(reward),
                next_state.astype(np.float32, copy=False),
                bool(done),
            )
        )

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )


@dataclass
class AgentConfig:
    gamma: float = 0.99
    tau: float = 0.01
    device: str = "cpu"


class DQNAgent:
    def __init__(
        self,
        policy_net: nn.Module,
        target_net: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: AgentConfig,
    ):
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.config = config
        self.loss_fn = nn.SmoothL1Loss()
        self.device = torch.device(config.device)

    def act(self, state: np.ndarray, epsilon: float, action_space_n: int) -> int:
        if random.random() < epsilon:
            return random.randrange(action_space_n)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self, batch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> float:
        states, actions, rewards, next_states, dones = batch
        device = self.device

        states_tensor = torch.from_numpy(states).to(device)
        actions_tensor = torch.from_numpy(actions).unsqueeze(1).to(device)
        rewards_tensor = torch.from_numpy(rewards).unsqueeze(1).to(device)
        next_states_tensor = torch.from_numpy(next_states).to(device)
        dones_tensor = torch.from_numpy(dones).unsqueeze(1).to(device)

        current_q = self.policy_net(states_tensor).gather(1, actions_tensor)
        with torch.no_grad():
            next_q = self.target_net(next_states_tensor).max(1, keepdim=True)[0]
        target_q = rewards_tensor + self.config.gamma * (1.0 - dones_tensor) * next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        self._soft_update()
        return float(loss.item())

    def _soft_update(self) -> None:
        tau = self.config.tau
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
