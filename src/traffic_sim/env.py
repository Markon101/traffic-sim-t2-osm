from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import SimulationConfig
from .road_network import EdgeKey
from .simulation import Simulation
from .traffic_signals import TrafficSignal


@dataclass
class EnvObservationSpace:
    max_signals: int = 48
    features_per_signal: int = 10


class TrafficEnv(gym.Env):
    """Gymnasium-compatible environment controlling coordinated signals."""

    metadata = {"render_modes": ["human", "headless"], "render_fps": 10}

    def __init__(
        self,
        config: SimulationConfig,
        *,
        control_interval_s: float = 5.0,
        observation_space_config: EnvObservationSpace | None = None,
    ):
        super().__init__()
        self.config = config
        self.control_interval_s = control_interval_s
        self.obs_config = observation_space_config or EnvObservationSpace()

        env_config = replace(config, headless=True, export_metrics=False)
        self.simulation = Simulation(env_config)
        self.signal_ids: List[int] = sorted(self.simulation.signals.keys())
        self.max_time_s = config.simulation_minutes * 60.0

        self._action_map = self._build_action_map()
        self.noop_action = 0
        self.action_space = spaces.Discrete(len(self._action_map) + 1)

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1e4,
            shape=(self.obs_config.max_signals, self.obs_config.features_per_signal),
            dtype=np.float32,
        )

        self.elapsed = 0.0

    # ------------------------------------------------------------------ #
    # Gym API                                                            #
    # ------------------------------------------------------------------ #

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        episode_seed = seed or np.random.randint(0, 1_000_000)
        self.simulation.reset(seed=int(episode_seed))
        self.elapsed = 0.0
        observation = self._encode_observation()
        info = {"time_s": self.simulation.time_s}
        return observation, info

    def step(self, action: int):
        metrics_before = self.simulation.metrics()

        override = None
        if action != self.noop_action:
            signal_id, phase_idx, duration = self._action_map[action - 1]
            override = {signal_id: (phase_idx, duration)}

        ticks = max(1, int(self.control_interval_s / self.config.dt))
        for _ in range(ticks):
            self.simulation.tick(override)

        metrics_after = self.simulation.metrics()
        reward = self._compute_reward(metrics_before, metrics_after)

        observation = self._encode_observation()
        self.elapsed = self.simulation.time_s
        terminated = self.elapsed >= self.max_time_s
        truncated = False
        info = {"time_s": self.simulation.time_s}
        return observation, reward, terminated, truncated, info

    def render(self):
        pass  # rendering handled by simulation visualiser

    def close(self) -> None:
        self.simulation.clear_overrides()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _build_action_map(self) -> List[Tuple[int, int, float]]:
        action_map: List[Tuple[int, int, float]] = []
        for signal_id in self.signal_ids:
            signal = self.simulation.signals[signal_id]
            for phase_idx, phase in enumerate(signal.phases):
                durations = [phase.min_duration_s, phase.base_duration_s, phase.max_duration_s]
                for duration in durations:
                    action_map.append((signal_id, phase_idx, duration))
        return action_map

    def _encode_observation(self) -> np.ndarray:
        max_signals = self.obs_config.max_signals
        feature_dim = self.obs_config.features_per_signal
        obs = np.zeros((max_signals, feature_dim), dtype=np.float32)

        queue_lengths = self.simulation.queue_lengths()
        edge_stats = self._edge_vehicle_stats()
        incident_edges = {
            incident.edge_key
            for incident in self.simulation.incident_manager.active_incidents.values()
        }

        for idx, signal_id in enumerate(self.signal_ids[:max_signals]):
            signal = self.simulation.signals[signal_id]
            features = self._signal_features(signal, queue_lengths, edge_stats, incident_edges)
            obs[idx, :] = features

        return obs

    def _signal_features(
        self,
        signal: TrafficSignal,
        queue_lengths: Dict[EdgeKey, float],
        edge_stats: Dict[EdgeKey, Tuple[int, float]],
        incident_edges: set[EdgeKey],
    ) -> np.ndarray:
        phases = signal.phases
        incoming = {edge for phase in phases for edge in phase.incoming_edges}
        if not incoming:
            return np.zeros(self.obs_config.features_per_signal, dtype=np.float32)

        queues = np.array([queue_lengths.get(edge, 0.0) for edge in incoming], dtype=np.float32)
        counts = np.array([edge_stats.get(edge, (0, 0.0))[0] for edge in incoming], dtype=np.float32)
        speeds = np.array([edge_stats.get(edge, (0, 0.0))[1] for edge in incoming], dtype=np.float32)
        lanes = np.array(
            [self.simulation.network.edges[edge].lanes for edge in incoming],
            dtype=np.float32,
        )

        total_queue = float(queues.sum())
        avg_queue = float(queues.mean()) if queues.size else 0.0
        max_queue = float(queues.max()) if queues.size else 0.0
        avg_speed = float(speeds.mean()) if speeds.size else 0.0
        density = float((counts / np.maximum(lanes, 1.0)).mean()) if counts.size else 0.0
        incident_flag = float(any(edge in incident_edges for edge in incoming))

        num_phases = float(len(phases))
        current_phase = float(signal.current_phase_index) / max(1.0, num_phases - 1)
        phase_progress = float(signal.time_in_phase / max(1.0, signal.target_duration))
        saturation = float(total_queue / max(1.0, queues.size))

        return np.array(
            [
                total_queue,
                avg_queue,
                max_queue,
                avg_speed,
                density,
                incident_flag,
                current_phase,
                phase_progress,
                num_phases,
                saturation,
            ],
            dtype=np.float32,
        )

    def _edge_vehicle_stats(self) -> Dict[EdgeKey, Tuple[int, float]]:
        stats: Dict[EdgeKey, Tuple[int, float]] = {}
        speed_accumulator: Dict[EdgeKey, float] = {}
        count_accumulator: Dict[EdgeKey, int] = {}

        for vehicle in self.simulation.vehicles:
            if vehicle.destination_reached():
                continue
            edge_key = vehicle.current_edge()
            count_accumulator[edge_key] = count_accumulator.get(edge_key, 0) + 1
            speed_accumulator[edge_key] = speed_accumulator.get(edge_key, 0.0) + vehicle.speed_mps

        for edge_key, count in count_accumulator.items():
            total_speed = speed_accumulator[edge_key]
            avg_speed = total_speed / max(1, count)
            stats[edge_key] = (count, avg_speed)

        return stats

    def _compute_reward(self, metrics_before: Dict[str, float], metrics_after: Dict[str, float]) -> float:
        queue_penalty = metrics_after.get("total_queue", 0.0)
        travel_delta = metrics_after.get("avg_travel_time_s", 0.0) - metrics_before.get("avg_travel_time_s", 0.0)
        fuel_delta = metrics_after.get("avg_fuel_l_per_trip", 0.0) - metrics_before.get("avg_fuel_l_per_trip", 0.0)
        completion_gain = metrics_after["completed_trips"] - metrics_before["completed_trips"]

        values = [queue_penalty, travel_delta, fuel_delta]
        values = [0.0 if not math.isfinite(val) else val for val in values]
        queue_penalty, travel_delta, fuel_delta = values

        reward = (
            -0.1 * queue_penalty
            -0.05 * travel_delta
            -0.4 * fuel_delta
            + 5.0 * completion_gain
        )
        return float(reward)
