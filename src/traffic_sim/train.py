from __future__ import annotations

import math
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
import typer
from gymnasium.vector import AsyncVectorEnv
from gymnasium.vector.async_vector_env import AutoresetMode

from .agent import AgentConfig, DQNAgent, ReplayBuffer
from .config import SimulationConfig
from .env import TrafficEnv
from .model import ModelConfig, build_model


app = typer.Typer(help="Training utilities for the traffic simulator.")


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@app.command()
def dqn(
    episodes: int = typer.Option(120, help="Number of training episodes."),
    warmup_steps: int = typer.Option(10_000, help="Steps before starting gradient updates."),
    batch_size: int = typer.Option(128, help="Mini-batch size for optimisation."),
    train_interval: int = typer.Option(1, help="Environment steps between optimisation rounds."),
    updates_per_step: int = typer.Option(2, help="Gradient updates to run each optimisation step."),
    replay_capacity: int = typer.Option(500_000, help="Replay buffer capacity."),
    gamma: float = typer.Option(0.97, help="Discount factor."),
    tau: float = typer.Option(0.01, help="Soft update coefficient for the target network."),
    lr: float = typer.Option(3e-4, help="AdamW learning rate."),
    epsilon_start: float = typer.Option(0.9, help="Initial epsilon for epsilon-greedy exploration."),
    epsilon_end: float = typer.Option(0.05, help="Final epsilon for epsilon-greedy."),
    epsilon_decay: int = typer.Option(100_000, help="Linear decay steps for epsilon."),
    model_type: str = typer.Option("transformer", help="Model architecture: transformer or residual."),
    model_width: int = typer.Option(256, help="Hidden size (d_model) for transformer models."),
    model_layers: int = typer.Option(6, help="Number of transformer encoder layers."),
    model_heads: int = typer.Option(8, help="Attention heads for transformer models."),
    model_dropout: float = typer.Option(0.1, help="Dropout applied within the transformer encoder."),
    control_interval: float = typer.Option(5.0, help="Seconds between RL actions."),
    output_dir: Path = typer.Option(Path("output/models"), help="Directory for model checkpoints."),
    num_envs: int = typer.Option(1, help="Number of parallel simulation environments."),
    device: str = typer.Option("auto", help="Training device: auto/cuda/cpu."),
    amp: bool = typer.Option(False, help="Use CUDA automatic mixed precision."),
    grad_clip: float = typer.Option(5.0, help="Gradient clipping norm."),
    seed: int = typer.Option(42, help="Random seed base value."),
) -> None:
    """Train a DQN-based signal controller against the simulator."""

    requested_device = device.lower()
    if requested_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = requested_device
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if resolved_device not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported device '{resolved_device}'. Expected auto/cuda/cpu.")
    device = resolved_device
    typer.echo(f"Using device: {device}")
    use_amp = bool(amp and device == "cuda")
    device_obj = torch.device(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = SimulationConfig(headless=True)
    probe_env = TrafficEnv(base_config, control_interval_s=control_interval)
    obs_shape = probe_env.observation_space.shape
    num_actions = probe_env.action_space.n
    probe_env.close()

    if model_type == "transformer" and model_width % model_heads != 0:
        raise ValueError("model_width must be divisible by model_heads for transformer models.")

    model_cfg = ModelConfig(
        input_dim=obs_shape[1],
        max_signals=obs_shape[0],
        num_actions=num_actions,
        d_model=model_width,
        num_layers=model_layers,
        num_heads=model_heads,
        dropout=model_dropout,
    )
    policy_net = build_model(model_type, model_cfg).to(device_obj)
    target_net = build_model(model_type, model_cfg).to(device_obj)
    target_net.load_state_dict(policy_net.state_dict())
    param_count = _count_parameters(policy_net)
    typer.echo(f"Model parameters: {param_count:,}")

    train_interval = max(1, train_interval)
    updates_per_step = max(1, updates_per_step)
    typer.echo(
        f"Optimiser schedule: interval={train_interval}, updates/step={updates_per_step}, "
        f"batch={batch_size}, warmup={warmup_steps}"
    )
    if use_amp:
        typer.echo("Automatic mixed precision (AMP) enabled.")

    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=lr, weight_decay=1e-4)
    agent = DQNAgent(
        policy_net,
        target_net,
        optimizer,
        AgentConfig(gamma=gamma, tau=tau, device=device, grad_clip=grad_clip, use_amp=use_amp),
    )
    buffer = ReplayBuffer(capacity=replay_capacity)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    global_step = 0
    epsilon = epsilon_start
    epsilon_decay_rate = (epsilon_start - epsilon_end) / max(1, epsilon_decay)
    best_reward = -math.inf

    rng = np.random.default_rng(seed)

    num_envs = max(1, num_envs)

    def _make_env_instance(env_seed: int) -> TrafficEnv:
        env_cfg = replace(base_config, random_seed=env_seed)
        return TrafficEnv(env_cfg, control_interval_s=control_interval)

    loss_accumulator: list[float] = []
    episodes_completed = 0

    if num_envs == 1:
        env_seed = int(rng.integers(0, 1_000_000))
        env = _make_env_instance(env_seed)
        try:
            state, _ = env.reset(seed=env_seed)
            episode_reward = 0.0
            episode_length = 0

            while episodes_completed < episodes:
                epsilon = max(epsilon_end, epsilon - epsilon_decay_rate)
                policy_net.eval()
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device_obj, non_blocking=True)
                    q_values = policy_net(state_tensor)
                policy_net.train()

                if float(rng.random()) < epsilon:
                    action = int(rng.integers(0, num_actions))
                else:
                    action = int(torch.argmax(q_values, dim=1).item())

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                buffer.push(state, action, reward, next_state, done)

                state = next_state
                episode_reward += float(reward)
                episode_length += 1
                global_step += 1

                if len(buffer) >= max(warmup_steps, batch_size) and global_step % train_interval == 0:
                    for _ in range(updates_per_step):
                        if len(buffer) < batch_size:
                            break
                        batch = buffer.sample(batch_size)
                        loss = agent.update(batch)
                        loss_accumulator.append(loss)

                if not done:
                    continue

                episodes_completed += 1
                mean_loss = float(np.mean(loss_accumulator)) if loss_accumulator else 0.0
                loss_accumulator.clear()

                gpu_extra = ""
                if device == "cuda":
                    torch.cuda.synchronize()
                    current_mem = torch.cuda.memory_allocated() / (1024 ** 3)
                    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    gpu_extra = f" | gpu-mem={current_mem:4.1f} GiB (peak {peak_mem:4.1f})"
                    torch.cuda.reset_peak_memory_stats()

                typer.echo(
                    f"Episode {episodes_completed:03d}/{episodes} | "
                    f"reward={episode_reward:8.2f} | "
                    f"steps={episode_length:04d} | "
                    f"epsilon={epsilon:.3f} | "
                    f"loss={mean_loss:.4f}"
                    f"{gpu_extra}"
                )

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    _save_checkpoint(
                        output_dir,
                        model_type,
                        policy_net,
                        optimizer,
                        episodes_completed - 1,
                        episode_reward,
                    )

                state, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
                episode_reward = 0.0
                episode_length = 0
        finally:
            env.close()
        return

    # Vectorised environment branch
    env_seeds = [seed + 997 * i for i in range(num_envs)]
    env_fns = [lambda s=s: _make_env_instance(s) for s in env_seeds]
    vec_env = AsyncVectorEnv(
        env_fns,
        shared_memory=True,
        autoreset_mode=AutoresetMode.NEXT_STEP,
    )

    try:
        states, _ = vec_env.reset(seed=env_seeds)
        episode_returns = np.zeros(num_envs, dtype=np.float32)
        episode_lengths = np.zeros(num_envs, dtype=np.int32)

        while episodes_completed < episodes:
            epsilon = max(epsilon_end, epsilon - epsilon_decay_rate * num_envs)

            policy_net.eval()
            with torch.no_grad():
                state_tensor = torch.from_numpy(states).to(device_obj, non_blocking=True)
                q_values = policy_net(state_tensor)
            policy_net.train()

            greedy_actions = torch.argmax(q_values, dim=1).cpu().numpy()
            random_mask = rng.random(num_envs) < epsilon
            random_actions = rng.integers(0, num_actions, size=num_envs)
            actions = greedy_actions
            actions[random_mask] = random_actions[random_mask]

            next_states, rewards, terminated, truncated, infos = vec_env.step(actions)
            dones = np.logical_or(terminated, truncated)

            for idx in range(num_envs):
                buffer.push(states[idx], int(actions[idx]), float(rewards[idx]), next_states[idx], bool(dones[idx]))

            states = next_states
            episode_returns += rewards
            episode_lengths += 1
            global_step += num_envs

            if len(buffer) >= max(warmup_steps, batch_size) and global_step % train_interval == 0:
                for _ in range(updates_per_step):
                    if len(buffer) < batch_size:
                        break
                    batch = buffer.sample(batch_size)
                    loss = agent.update(batch)
                    loss_accumulator.append(loss)

            mean_loss_for_log = float(np.mean(loss_accumulator)) if loss_accumulator else 0.0
            logged_any = False
            for idx in range(num_envs):
                if not dones[idx]:
                    continue

                episodes_completed += 1
                ep_reward = float(episode_returns[idx])
                ep_steps = int(episode_lengths[idx])
                logged_any = True

                gpu_extra = ""
                if device == "cuda":
                    torch.cuda.synchronize()
                    current_mem = torch.cuda.memory_allocated() / (1024 ** 3)
                    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    gpu_extra = f" | gpu-mem={current_mem:4.1f} GiB (peak {peak_mem:4.1f})"
                    torch.cuda.reset_peak_memory_stats()

                typer.echo(
                    f"Episode {episodes_completed:03d}/{episodes} | "
                    f"reward={ep_reward:8.2f} | "
                    f"steps={ep_steps:04d} | "
                    f"epsilon={epsilon:.3f} | "
                    f"loss={mean_loss_for_log:.4f}"
                    f"{gpu_extra}"
                )

                if ep_reward > best_reward:
                    best_reward = ep_reward
                    _save_checkpoint(
                        output_dir,
                        model_type,
                        policy_net,
                        optimizer,
                        episodes_completed - 1,
                        ep_reward,
                    )

                episode_returns[idx] = 0.0
                episode_lengths[idx] = 0

                if episodes_completed >= episodes:
                    break

            if logged_any:
                loss_accumulator.clear()

    finally:
        vec_env.close()


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
