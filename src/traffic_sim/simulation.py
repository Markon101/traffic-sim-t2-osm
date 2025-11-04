from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from .config import DEFAULT_VEHICLE_PROFILES, SimulationConfig, VehicleProfile
from .controllers import AdaptiveHeuristicController, TrafficController
from .events import Incident, IncidentManager
from .osm_loader import load_graph
from .road_network import EdgeKey, RoadNetwork
from .traffic_signals import TrafficSignal, build_signals
from .vehicle import Vehicle
from .visualization import PygameVisualizer

LOGGER = logging.getLogger(__name__)


@dataclass
class SimulationReport:
    time_s: float
    active_vehicles: int
    completed_trips: int
    avg_speed_mps: float
    avg_fuel_l_per_trip: float
    avg_travel_time_s: float
    total_queue: float
    incidents: List[Incident] = field(default_factory=list)


class Simulation:
    """Core simulation loop coordinating agents, signals, and incidents."""

    def __init__(
        self,
        config: SimulationConfig,
        *,
        controller: Optional[TrafficController] = None,
        vehicle_profiles: Optional[Dict[str, VehicleProfile]] = None,
        visualizer: Optional[PygameVisualizer] = None,
        force_refresh_graph: bool = False,
    ):
        self.config = config
        self.controller = controller
        self.vehicle_profiles = vehicle_profiles or DEFAULT_VEHICLE_PROFILES
        self.visualizer = visualizer

        self.graph = load_graph(config, force_refresh=force_refresh_graph)
        self.network = RoadNetwork(self.graph)
        if self.controller is None:
            self.controller = AdaptiveHeuristicController(self.network)

        self.incident_manager = IncidentManager(
            self.network,
            hourly_rate=config.incident_rate_per_hour,
            rng=config.rng(),
        )

        self.signals: Dict[int, TrafficSignal] = build_signals(
            self.network, yellow_duration_s=config.signal_yellow_duration
        )
        self.rng = config.rng()
        self.control_warmup_s = max(0.0, config.signal_warmup_seconds)
        self._warmup_complete = self.control_warmup_s <= 0.0

        self.vehicles: List[Vehicle] = []
        self.completed: List[Vehicle] = []
        self.time_s: float = 0.0
        self._next_vehicle_id = 1
        self._queue_lengths: Dict[EdgeKey, float] = {}
        self._overrides: Dict[int, Tuple[int, float]] = {}
        self._latest_incidents: List[Incident] = []
        self._metrics_header_written = False

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.incident_manager.rng = np.random.default_rng(seed + 1)
        self.vehicles.clear()
        self.completed.clear()
        self.time_s = 0.0
        self._next_vehicle_id = 1
        self._queue_lengths.clear()
        self._overrides.clear()
        self._latest_incidents.clear()
        self._metrics_header_written = False
        self._warmup_complete = self.control_warmup_s <= 0.0
        for edge in self.network.iter_edges():
            if edge.is_closed:
                self.network.open_edge(edge.key)
        for signal in self.signals.values():
            signal.current_phase_index = 0
            signal.time_in_phase = 0.0
            signal.target_duration = signal.current_phase().base_duration_s
            signal.history.clear()
            signal.in_transition = False
            signal.pending_phase_index = None
            signal.pending_target_duration = None

    def run(self) -> SimulationReport:
        steps = self.config.total_steps()
        for _ in range(steps):
            if not self.tick():
                break
        return self._compile_report()

    def tick(self, overrides: Optional[Dict[int, Tuple[int, float]]] = None) -> bool:
        """Advance the simulation by one timestep."""

        dt = self.config.dt
        if overrides:
            self._overrides.update(overrides)

        self._spawn_vehicles(dt)
        edge_vehicle_map = self._group_vehicles_by_edge()
        self._queue_lengths = self._estimate_queue_lengths(edge_vehicle_map)

        self._update_signals(dt)
        self._latest_incidents = self.incident_manager.update(self.time_s, dt)

        self._update_vehicles(edge_vehicle_map, dt)

        self.time_s += dt
        if self.config.export_metrics:
            self._write_metrics_row()

        if not self.config.headless and self.visualizer:
            if not self.visualizer.handle_events():
                return False
            metrics = self._metrics_snapshot()
            incidents = {
                incident.edge_key: incident
                for incident in self.incident_manager.active_incidents.values()
            }
            self.visualizer.draw(self.vehicles, self.signals, incidents, metrics)
            self.visualizer.tick()

        return True

    def apply_override(self, signal_id: int, phase_index: int, duration: float) -> None:
        self._overrides[signal_id] = (phase_index, duration)

    def clear_overrides(self) -> None:
        self._overrides.clear()

    def queue_lengths(self) -> Dict[EdgeKey, float]:
        return dict(self._queue_lengths)

    def metrics(self) -> Dict[str, float]:
        return self._metrics_snapshot()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _spawn_vehicles(self, dt: float) -> None:
        expected = self.config.spawn_rate_per_minute * dt / 60.0
        spawn_count = self.rng.poisson(expected)
        if not spawn_count:
            return

        if len(self.vehicles) >= self.config.max_vehicles:
            return

        nodes = list(self.network.nodes.keys())
        weights = np.array([self.network.nodes[n].street_count for n in nodes], dtype=float)
        weights = weights / weights.sum()

        vehicle_mix = list(self.config.normalized_vehicle_mix().items())
        vehicle_types, vehicle_prob = zip(*vehicle_mix)

        for _ in range(spawn_count):
            if len(self.vehicles) >= self.config.max_vehicles:
                break
            origin = int(self.rng.choice(nodes, p=weights))
            destination = int(self.rng.choice(nodes, p=weights))
            if origin == destination:
                continue

            route_nodes = self._safe_route(origin, destination)
            if not route_nodes or len(route_nodes) < 2:
                continue

            route_edges = self._nodes_to_edges(route_nodes)
            if not route_edges:
                continue

            profile_name = str(self.rng.choice(vehicle_types, p=vehicle_prob))
            profile = self.vehicle_profiles.get(profile_name)
            if profile is None:
                profile = DEFAULT_VEHICLE_PROFILES["sedan"]

            vehicle = Vehicle(
                vehicle_id=self._next_vehicle_id,
                profile=profile,
                route_nodes=route_nodes,
                route_edges=route_edges,
                spawn_time=self.time_s,
                destination=route_nodes[-1],
                speed_mps=0.0,
            )
            self.vehicles.append(vehicle)
            self._next_vehicle_id += 1

    def _safe_route(self, origin: int, destination: int) -> Optional[List[int]]:
        try:
            return self.network.shortest_path(origin, destination)
        except (nx.NetworkXNoPath, nx.NodeNotFound):  # pragma: no cover - extremely rare
            return None

    def _nodes_to_edges(self, nodes: List[int]) -> List[EdgeKey]:
        edges: List[EdgeKey] = []
        for u, v in zip(nodes[:-1], nodes[1:]):
            candidates = [
                key for key in self.network.graph[u][v].keys()
                if not self.network.is_edge_closed((u, v, key))
            ]
            if not candidates:
                return []
            best_key = min(
                candidates,
                key=lambda k: self.network.graph[u][v][k].get("travel_time", math.inf),
            )
            edges.append((u, v, best_key))
        return edges

    def _group_vehicles_by_edge(self) -> Dict[EdgeKey, List[Vehicle]]:
        edge_map: Dict[EdgeKey, List[Vehicle]] = {}
        for vehicle in self.vehicles:
            if vehicle.destination_reached():
                continue
            edge_key = vehicle.current_edge()
            edge_map.setdefault(edge_key, []).append(vehicle)
        for vehicles in edge_map.values():
            vehicles.sort(key=lambda v: v.distance_on_edge, reverse=True)
        return edge_map

    def _estimate_queue_lengths(self, edge_map: Dict[EdgeKey, List[Vehicle]]) -> Dict[EdgeKey, float]:
        queues: Dict[EdgeKey, float] = {}
        for edge_key, vehicles in edge_map.items():
            queue_value = 0.0
            for vehicle in vehicles:
                distance_to_end = vehicle.distance_to_edge_end(self.network)
                if distance_to_end < 18.0 and vehicle.speed_mps < 2.0:
                    queue_value += 1.0 + vehicle.profile.length_m / 12.0
            if queue_value > 0:
                queues[edge_key] = queue_value
        return queues

    def _update_signals(self, dt: float) -> None:
        adaptive_enabled = self._warmup_complete or self.time_s >= self.control_warmup_s
        if not self._warmup_complete and adaptive_enabled:
            self._warmup_complete = True
            self._overrides.clear()

        active_overrides = self._overrides if adaptive_enabled else {}
        next_overrides: Dict[int, Tuple[int, float]] = {} if adaptive_enabled else self._overrides

        for node_id, signal in self.signals.items():
            all_edges = {edge for phase in signal.phases for edge in phase.incoming_edges}
            localized_queue = {edge: self._queue_lengths.get(edge, 0.0) for edge in all_edges}

            if adaptive_enabled and node_id in active_overrides:
                phase_idx, duration = active_overrides[node_id]
                signal.set_phase(phase_idx, duration)
                next_overrides[node_id] = (phase_idx, duration)
            elif adaptive_enabled and self.controller:
                decision = self.controller.decide(self.time_s, signal, localized_queue)
                if decision:
                    signal.set_phase(*decision)

            signal.advance(dt, localized_queue, adaptive=adaptive_enabled)

        if adaptive_enabled:
            self._overrides = next_overrides

    def _update_vehicles(self, edge_map: Dict[EdgeKey, List[Vehicle]], dt: float) -> None:
        completed: List[Vehicle] = []
        for edge_key, vehicles in edge_map.items():
            leader: Optional[Vehicle] = None
            signal_green = self._signal_for_edge(edge_key)
            for vehicle in vehicles:
                vehicle.update(dt, self.network, leader, signal_green)
                if vehicle.destination_reached():
                    completed.append(vehicle)
                    leader = None
                    continue
                if vehicle.current_edge() != edge_key:
                    leader = None
                else:
                    leader = vehicle

        if completed:
            completed_ids = {vehicle.vehicle_id for vehicle in completed}
            self.vehicles = [v for v in self.vehicles if v.vehicle_id not in completed_ids]
            self.completed.extend(completed)

    def _signal_for_edge(self, edge_key: EdgeKey) -> bool:
        _, dest, _ = edge_key
        signal = self.signals.get(dest)
        if signal is None:
            return True
        return signal.is_green(edge_key)

    def _metrics_snapshot(self) -> Dict[str, float]:
        if self.vehicles:
            avg_speed = float(np.mean([vehicle.speed_mps for vehicle in self.vehicles]))
        else:
            avg_speed = 0.0

        if self.completed:
            avg_travel = float(np.mean([v.travel_time_s for v in self.completed]))
            avg_fuel = float(np.mean([v.fuel_consumed_l for v in self.completed]))
        else:
            avg_travel = 0.0
            avg_fuel = 0.0

        metrics = {
            "time_s": self.time_s,
            "active_vehicles": len(self.vehicles),
            "completed_trips": len(self.completed),
            "avg_speed_mps": avg_speed,
            "avg_travel_time_s": avg_travel,
            "avg_fuel_l_per_trip": avg_fuel,
            "total_queue": float(sum(self._queue_lengths.values())),
        }
        return metrics

    def _compile_report(self) -> SimulationReport:
        metrics = self._metrics_snapshot()
        return SimulationReport(
            time_s=metrics["time_s"],
            active_vehicles=metrics["active_vehicles"],
            completed_trips=metrics["completed_trips"],
            avg_speed_mps=metrics["avg_speed_mps"],
            avg_fuel_l_per_trip=metrics["avg_fuel_l_per_trip"],
            avg_travel_time_s=metrics["avg_travel_time_s"],
            total_queue=metrics["total_queue"],
            incidents=list(self.incident_manager.active_incidents.values()),
        )

    def _write_metrics_row(self) -> None:
        metrics_path: Path = self.config.metrics_output
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = self._metrics_snapshot()
        columns = [
            "time_s",
            "active_vehicles",
            "completed_trips",
            "avg_speed_mps",
            "avg_travel_time_s",
            "avg_fuel_l_per_trip",
            "total_queue",
        ]
        line = ",".join(str(metrics[col]) for col in columns)
        if not self._metrics_header_written or not metrics_path.exists():
            header = ",".join(columns)
            with metrics_path.open("w", encoding="utf-8") as fp:
                fp.write(header + "\n")
                fp.write(line + "\n")
            self._metrics_header_written = True
        else:
            with metrics_path.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")
