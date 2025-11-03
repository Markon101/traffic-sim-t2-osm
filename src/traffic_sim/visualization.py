from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import numpy as np
from shapely.geometry import LineString

from .config import SimulationConfig
from .road_network import EdgeKey, RoadEdge, RoadNetwork
from .traffic_signals import TrafficSignal
from .utils import compute_transform, line_to_points, world_to_screen
from .vehicle import Vehicle

try:
    import pygame
except Exception:  # pragma: no cover
    pygame = None


class PygameVisualizer:
    """Real-time pygame visualisation of the traffic network."""

    def __init__(self, network: RoadNetwork, config: SimulationConfig):
        if pygame is None:
            raise ImportError("pygame is required for visualisation mode")

        pygame.init()
        pygame.display.set_caption("Traffic Simulator")
        self.screen = pygame.display.set_mode(config.screen_size)
        self.clock = pygame.time.Clock()
        self.config = config
        self.network = network
        self.scale, self.translation = compute_transform(network.bounds(), config.screen_size)
        self.font = pygame.font.SysFont("Arial", 16)
        self.road_sprites = self._create_road_sprites()

    def close(self) -> None:
        if pygame:
            pygame.quit()

    def _create_road_sprites(self) -> Dict[EdgeKey, np.ndarray]:
        sprites: Dict[EdgeKey, np.ndarray] = {}
        for edge in self.network.iter_edges():
            xs, ys = line_to_points(
                edge.geometry.coords,
                self.scale,
                self.translation,
                screen_height=self.config.screen_size[1],
            )
            sprites[edge.key] = np.stack([xs, ys], axis=1)
        return sprites

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def draw(
        self,
        vehicles: Iterable[Vehicle],
        signals: Dict[int, TrafficSignal],
        incidents: Dict[int, float],
        metrics: Dict[str, float],
    ) -> None:
        self.screen.fill((16, 20, 32))
        self._draw_roads(incidents)
        self._draw_signals(signals)
        self._draw_vehicles(vehicles)
        self._draw_overlay(metrics)
        pygame.display.flip()

    def _draw_roads(self, incidents: Dict[int, float]) -> None:
        for edge_key, sprite in self.road_sprites.items():
            if not len(sprite):
                continue
            width = 2
            edge = self.network.edges[edge_key]
            if edge.is_closed:
                color = (200, 60, 60)
                width = 3
            else:
                color = (60, 100, 140)
            pygame.draw.lines(self.screen, color, False, sprite, width)

    def _draw_signals(self, signals: Dict[int, TrafficSignal]) -> None:
        for node_id, signal in signals.items():
            node = self.network.nodes[node_id]
            pos = world_to_screen(
                (node.x, node.y),
                self.scale,
                self.translation,
                screen_height=self.config.screen_size[1],
            )
            active_phase = signal.current_phase()
            active_color = (80, 220, 80)
            inactive_color = (40, 40, 40)
            pygame.draw.circle(self.screen, inactive_color, pos, 10)
            pygame.draw.circle(self.screen, active_color, pos, 6, width=0)
            label = self.font.render(active_phase.name[:8], True, (230, 230, 230))
            self.screen.blit(label, (pos[0] + 8, pos[1] - 8))

    def _draw_vehicles(self, vehicles: Iterable[Vehicle]) -> None:
        for vehicle in vehicles:
            if vehicle.destination_reached():
                continue
            edge = self.network.edges[vehicle.current_edge()]
            point = _point_along(edge.geometry, vehicle.distance_on_edge)
            pos = world_to_screen(
                (point[0], point[1]),
                self.scale,
                self.translation,
                screen_height=self.config.screen_size[1],
            )
            size = max(3, int(vehicle.profile.length_m / 1.5))
            color = _vehicle_color(vehicle.profile.name)
            pygame.draw.circle(self.screen, color, pos, size)

    def _draw_overlay(self, metrics: Dict[str, float]) -> None:
        lines = [
            f"Vehicles: {int(metrics.get('active_vehicles', 0))}",
            f"Completed: {int(metrics.get('completed_trips', 0))}",
            f"Avg speed: {metrics.get('avg_speed_mps', 0):.1f} m/s",
            f"Avg fuel: {metrics.get('avg_fuel_l_per_trip', 0):.2f} L",
            f"Travel time idx: {metrics.get('avg_travel_time_s', 0):.0f} s",
        ]
        y = 12
        for text in lines:
            label = self.font.render(text, True, (240, 240, 240))
            self.screen.blit(label, (12, y))
            y += 18

    def tick(self, fps: int = 60) -> None:
        self.clock.tick(fps)


def _point_along(geometry: LineString, distance: float) -> np.ndarray:
    if geometry.length <= 0:
        x, y = geometry.coords[0]
        return np.array([x, y])
    clipped_distance = max(0.0, min(distance, geometry.length))
    point = geometry.interpolate(clipped_distance)
    return np.array([point.x, point.y])


def _vehicle_color(name: str) -> tuple[int, int, int]:
    palette = {
        "sedan": (90, 180, 255),
        "suv": (255, 140, 105),
        "delivery_van": (255, 220, 120),
        "semi": (210, 120, 255),
    }
    return palette.get(name, (200, 200, 200))
