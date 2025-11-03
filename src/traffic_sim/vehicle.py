from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .config import VehicleProfile
from .road_network import EdgeKey, RoadNetwork


@dataclass
class Vehicle:
    vehicle_id: int
    profile: VehicleProfile
    route_nodes: List[int]
    route_edges: List[EdgeKey]
    spawn_time: float
    destination: int
    current_edge_index: int = 0
    distance_on_edge: float = 0.0
    speed_mps: float = 0.0
    travel_time_s: float = 0.0
    fuel_consumed_l: float = 0.0
    waiting_for_signal: bool = False
    reroute_requested: bool = False
    history: List[Tuple[float, EdgeKey, float]] = field(default_factory=list)

    def current_edge(self) -> EdgeKey:
        return self.route_edges[self.current_edge_index]

    def destination_reached(self) -> bool:
        return self.current_edge_index >= len(self.route_edges)

    def update(
        self,
        dt: float,
        network: RoadNetwork,
        leader: Optional["Vehicle"],
        signal_green: bool,
    ) -> None:
        """Advance state using a simple car-following model."""

        if self.destination_reached():
            return

        edge = network.edges[self.current_edge()]
        target_speed = min(self.profile.max_speed_mps, edge.speed_mps)

        # Reduce speed if the signal ahead is red
        if not signal_green and self.distance_to_edge_end(network) < 10.0:
            target_speed = min(target_speed, 2.0)

        # Simple IDM-inspired spacing with the leader vehicle
        if leader and leader.current_edge() == self.current_edge():
            gap = leader.distance_on_edge - self.distance_on_edge - leader.profile.length_m
            if gap < 5.0:
                target_speed = min(target_speed, max(0.0, leader.speed_mps - 2.0))

        acceleration = (target_speed - self.speed_mps) / max(dt, 1e-3)
        acceleration = max(-self.profile.acceleration_mps2, min(self.profile.acceleration_mps2, acceleration))

        self.speed_mps = max(0.0, self.speed_mps + acceleration * dt)
        move_distance = self.speed_mps * dt

        remaining = edge.length_m - self.distance_on_edge
        if move_distance >= remaining - 1e-2:
            move_distance = remaining
            self.distance_on_edge = edge.length_m
        else:
            self.distance_on_edge += move_distance

        self.travel_time_s += dt
        self._update_fuel(acceleration, dt)
        self.waiting_for_signal = self.speed_mps < 0.5 and move_distance < 0.1

        if self.distance_on_edge >= edge.length_m - 1e-2:
            self._advance_edge(network, dt)

    def _advance_edge(self, network: RoadNetwork, dt: float) -> None:
        edge_key = self.current_edge()
        self.history.append((self.travel_time_s, edge_key, self.distance_on_edge))
        self.current_edge_index += 1
        if self.destination_reached():
            return
        self.distance_on_edge = 0.0

    def distance_to_edge_end(self, network: RoadNetwork) -> float:
        if self.destination_reached():
            return 0.0
        edge = network.edges[self.current_edge()]
        return max(0.0, edge.length_m - self.distance_on_edge)

    def _update_fuel(self, acceleration: float, dt: float) -> None:
        base_rate = self.profile.base_fuel_consumption_l_per_100km / 100.0
        dynamic_penalty = abs(acceleration) * self.profile.fuel_penalty_per_mps2
        speed_penalty = max(0.0, (self.speed_mps - 12.0) / 12.0)
        consumption = (base_rate + dynamic_penalty + speed_penalty) * self.speed_mps * dt
        self.fuel_consumed_l += consumption

    def request_reroute(self) -> None:
        self.reroute_requested = True
