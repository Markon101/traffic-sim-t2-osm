from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer

from .agent import AgentConfig, DQNAgent, ReplayBuffer
from .config import SimulationConfig
from .env import TrafficEnv
from .model import ModelConfig, build_model


app = typer.Typer(help="Training utilities for the traffic simulator.")


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@app.command()
def dqn(
    episodes: int = typer.Option(50, help="Number of training episodes."),
    warmup_steps: int = typer.Option(2_000, help="Steps before starting gradient updates."),
    batch_size: int = typer.Option(64, help="Mini-batch size for optimisation."),
    gamma: float = typer.Option(0.97, help="Discount factor."),
    tau: float = typer.Option(0.01, help="Soft update coefficient for the target network."),
    lr: float = typer.Option(3e-4, help="AdamW learning rate."),
    epsilon_start: float = typer.Option(0.9, help="Initial epsilon for epsilon-greedy exploration."),
    epsilon_end: float = typer.Option(0.05, help="Final epsilon for epsilon-greedy."),
    epsilon_decay: int = typer.Option(50_000, help="Linear decay steps for epsilon."),
    model_type: str = typer.Option("transformer", help="Model architecture: transformer or residual."),
    control_interval: float = typer.Option(5.0, help="Seconds between RL actions."),
    output_dir: Path = typer.Option(Path("output/models"), help="Directory for model checkpoints."),
    seed: int = typer.Option(42, help="Random seed base value."),
) -> None:
    """Train a DQN-based signal controller against the simulator."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    typer.echo(f"Using device: {device}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = SimulationConfig(headless=True)
    env = TrafficEnv(base_config, control_interval_s=control_interval)
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n

    model_cfg = ModelConfig(
        input_dim=obs_shape[1],
        max_signals=obs_shape[0],
        num_actions=num_actions,
    )
    policy_net = build_model(model_type, model_cfg).to(device)
    target_net = build_model(model_type, model_cfg).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    param_count = _count_parameters(policy_net)
    typer.echo(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=lr, weight_decay=1e-4)
    agent = DQNAgent(
        policy_net,
        target_net,
        optimizer,
        AgentConfig(gamma=gamma, tau=tau, device=device),
    )
    buffer = ReplayBuffer(capacity=200_000)

    global_step = 0
    epsilon = epsilon_start
    epsilon_decay_rate = (epsilon_start - epsilon_end) / max(1, epsilon_decay)
    best_reward = -math.inf

    rng = np.random.default_rng(seed)

    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
            done = False
            episode_reward = 0.0
            episode_losses: list[float] = []
            steps = 0

            while not done:
                epsilon = max(epsilon_end, epsilon - epsilon_decay_rate)
                action = agent.act(state, epsilon, num_actions)
                next_state, reward, terminated, truncated, _ = env.step(action)

                done = terminated or truncated
                buffer.push(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                steps += 1
                global_step += 1

                if len(buffer) >= warmup_steps and global_step % 4 == 0:
                    batch = buffer.sample(batch_size)
                    loss = agent.update(batch)
                    episode_losses.append(loss)

            mean_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
            typer.echo(
                f"Episode {episode + 1:03d}/{episodes} | "
                f"reward={episode_reward:8.2f} | "
                f"steps={steps:04d} | "
                f"epsilon={epsilon:.3f} | "
                f"loss={mean_loss:.4f}"
            )

            if episode_reward > best_reward:
                best_reward = episode_reward
                _save_checkpoint(
                    output_dir,
                    model_type,
                    policy_net,
                    optimizer,
                    episode,
                    episode_reward,
                )
    finally:
        env.close()


def _save_checkpoint(
    output_dir: Path,
    model_type: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    episode: int,
    score: float,
) -> None:
    path = output_dir / f"{model_type}_dqn.pt"
    torch.save(
        {
            "model_type": model_type,
            "episode": episode,
            "score": score,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    typer.echo(f"Saved checkpoint to {path}")
