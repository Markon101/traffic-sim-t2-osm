from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .road_network import EdgeKey, RoadNetwork


@dataclass
class SignalPhase:
    name: str
    incoming_edges: List[EdgeKey]
    min_duration_s: float = 12.0
    max_duration_s: float = 45.0
    base_duration_s: float = 20.0


@dataclass
class TrafficSignal:
    node_id: int
    phases: List[SignalPhase]
    current_phase_index: int = 0
    time_in_phase: float = 0.0
    target_duration: float = 20.0
    history: List[Tuple[float, int]] = field(default_factory=list)
    yellow_duration_s: float = 3.0
    in_transition: bool = False
    pending_phase_index: Optional[int] = None
    pending_target_duration: Optional[float] = None

    def current_phase(self) -> SignalPhase:
        return self.phases[self.current_phase_index]

    def is_green(self, edge_key: EdgeKey) -> bool:
        if self.in_transition:
            return False
        return edge_key in self.current_phase().incoming_edges

    def is_transitioning(self) -> bool:
        return self.in_transition

    def transition_progress(self) -> float:
        if not self.in_transition or self.yellow_duration_s <= 0:
            return 0.0
        return float(min(1.0, self.time_in_phase / self.yellow_duration_s))

    def advance(self, dt: float, queue_lengths: Dict[EdgeKey, float], adaptive: bool = True) -> None:
        self.time_in_phase += dt
        if self.in_transition:
            if self.time_in_phase >= self.yellow_duration_s:
                self._complete_transition()
            return

        current_phase = self.current_phase()
        if self.time_in_phase < self.target_duration:
            return

        if not adaptive:
            next_idx = (self.current_phase_index + 1) % len(self.phases)
            next_duration = self.phases[next_idx].base_duration_s
            self._begin_transition(next_idx, next_duration)
            return

        pressures: List[Tuple[int, float]] = []
        for idx, phase in enumerate(self.phases):
            pressure = sum(queue_lengths.get(edge, 0.0) for edge in phase.incoming_edges)
            pressures.append((idx, pressure))
        pressures.sort(key=lambda item: item[1], reverse=True)
        next_idx = pressures[0][0] if pressures else (self.current_phase_index + 1) % len(self.phases)

        max_exceeded = self.time_in_phase >= current_phase.max_duration_s - 1e-6
        if max_exceeded and len(self.phases) > 1:
            next_idx = (self.current_phase_index + 1) % len(self.phases)
        elif pressures:
            top_pressure = pressures[0][1]
            tied_indices = [idx for idx, value in pressures if abs(value - top_pressure) < 1e-6]
            if next_idx == self.current_phase_index and len(tied_indices) > 1:
                if self.current_phase_index in tied_indices:
                    current_pos = tied_indices.index(self.current_phase_index)
                    next_idx = tied_indices[(current_pos + 1) % len(tied_indices)]
                else:
                    next_idx = tied_indices[0]

        if next_idx == self.current_phase_index:
            self.target_duration = min(current_phase.max_duration_s, self.target_duration + 5.0)
            return

        next_duration = self._duration_for_phase_index(next_idx, queue_lengths)
        self._begin_transition(next_idx, next_duration)

    def set_phase(self, phase_index: int, duration: float) -> None:
        duration = self._clamp_duration(phase_index, duration)
        if self.in_transition:
            self.pending_phase_index = phase_index
            self.pending_target_duration = duration
            return
        if phase_index == self.current_phase_index:
            self.target_duration = duration
        else:
            self._begin_transition(phase_index, duration)

    def _begin_transition(self, next_idx: int, next_duration: float) -> None:
        if next_idx == self.current_phase_index:
            self.target_duration = next_duration
            return
        self.history.append((self.time_in_phase, self.current_phase_index))
        self.pending_phase_index = next_idx
        self.pending_target_duration = next_duration
        self.in_transition = True
        self.time_in_phase = 0.0

    def _complete_transition(self) -> None:
        if self.pending_phase_index is not None:
            self.current_phase_index = self.pending_phase_index
        self.pending_phase_index = None
        self.in_transition = False
        self.time_in_phase = 0.0
        if self.pending_target_duration is not None:
            self.target_duration = self.pending_target_duration
        else:
            self.target_duration = self.current_phase().base_duration_s
        self.pending_target_duration = None

    def _duration_for_phase_index(self, phase_index: int, queue_lengths: Dict[EdgeKey, float]) -> float:
        phase = self.phases[phase_index]
        total_queue = sum(queue_lengths.get(edge, 0.0) for edge in phase.incoming_edges)
        scaled = phase.base_duration_s + total_queue * 0.5
        return float(min(max(phase.min_duration_s, scaled), phase.max_duration_s))

    def _clamp_duration(self, phase_index: int, duration: float) -> float:
        phase = self.phases[phase_index]
        return float(min(max(phase.min_duration_s, duration), phase.max_duration_s))


def build_signals(network: RoadNetwork, yellow_duration_s: float = 3.0) -> Dict[int, TrafficSignal]:
    signals: Dict[int, TrafficSignal] = {}
    for node_id, node in network.nodes.items():
        if node.street_count < 3:
            continue

        incoming = network.incoming_edges(node_id)
        if len(incoming) < 2:
            continue

        phases = _group_edges_into_phases(incoming, node_id, network)
        if len(phases) < 2:
            continue

        signal = TrafficSignal(node_id=node_id, phases=phases, yellow_duration_s=yellow_duration_s)
        signal.target_duration = phases[0].base_duration_s
        signals[node_id] = signal

    return signals


def _group_edges_into_phases(edges, node_id: int, network: RoadNetwork) -> List[SignalPhase]:
    buckets: Dict[str, List[EdgeKey]] = {
        "northbound": [],
        "eastbound": [],
        "southbound": [],
        "westbound": [],
    }

    for edge in edges:
        start_node = edge.key[0]
        x0, y0 = network.node_coordinates(start_node)
        x1, y1 = network.node_coordinates(node_id)
        angle = math.atan2(y1 - y0, x1 - x0)
        direction = _direction_bucket(angle)
        buckets[direction].append(edge.key)

    phases: List[SignalPhase] = []
    if buckets["northbound"] or buckets["southbound"]:
        phases.append(
            SignalPhase(
                name="north_south",
                incoming_edges=buckets["northbound"] + buckets["southbound"],
                base_duration_s=22.0,
            )
        )
    if buckets["eastbound"] or buckets["westbound"]:
        phases.append(
            SignalPhase(
                name="east_west",
                incoming_edges=buckets["eastbound"] + buckets["westbound"],
                base_duration_s=22.0,
            )
        )

    covered = {edge_key for phase in phases for edge_key in phase.incoming_edges}
    remaining = [edge.key for edge in edges if edge.key not in covered]
    if remaining:
        phases.append(
            SignalPhase(
                name="mixed",
                incoming_edges=remaining,
                base_duration_s=18.0,
            )
        )

    return phases


def _direction_bucket(angle: float) -> str:
    deg = math.degrees(angle) % 360
    if 45 <= deg < 135:
        return "northbound"
    if 135 <= deg < 225:
        return "westbound"
    if 225 <= deg < 315:
        return "southbound"
    return "eastbound"
