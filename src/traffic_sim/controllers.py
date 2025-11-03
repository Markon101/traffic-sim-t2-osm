from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .traffic_signals import SignalPhase, TrafficSignal
from .road_network import EdgeKey, RoadNetwork


class TrafficController:
    """Interface for external traffic control logic."""

    def decide(
        self,
        current_time: float,
        signal: TrafficSignal,
        queue_lengths: Dict[EdgeKey, float],
    ) -> Optional[Tuple[int, float]]:
        raise NotImplementedError


@dataclass
class AdaptiveHeuristicController(TrafficController):
    network: RoadNetwork

    def decide(
        self,
        current_time: float,
        signal: TrafficSignal,
        queue_lengths: Dict[EdgeKey, float],
    ) -> Optional[Tuple[int, float]]:
        best_phase = signal.current_phase_index
        best_pressure = -1.0

        for idx, phase in enumerate(signal.phases):
            pressure = _phase_pressure(phase, queue_lengths)
            if pressure > best_pressure:
                best_pressure = pressure
                best_phase = idx

        duration = _suggest_duration(signal.phases[best_phase], queue_lengths)
        return best_phase, duration


def _phase_pressure(phase: SignalPhase, queue_lengths: Dict[EdgeKey, float]) -> float:
    return sum(queue_lengths.get(edge, 0.0) for edge in phase.incoming_edges)


def _suggest_duration(phase: SignalPhase, queue_lengths: Dict[EdgeKey, float]) -> float:
    demand = _phase_pressure(phase, queue_lengths)
    duration = phase.base_duration_s + demand * 0.4
    duration = min(max(duration, phase.min_duration_s), phase.max_duration_s)
    return duration
