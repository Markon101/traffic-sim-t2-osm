from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from shapely.geometry import LineString

from .config import SimulationConfig
from .events import Incident
from .road_network import EdgeKey, RoadEdge, RoadNetwork
from .traffic_signals import TrafficSignal
from .utils import compute_transform
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
        self.config = config
        self.network = network
        self.screen = pygame.display.set_mode(config.screen_size)
        self.screen_rect = self.screen.get_rect()
        self.clock = pygame.time.Clock()

        scale, translation = compute_transform(network.bounds(), config.screen_size)
        self.default_scale = float(scale)
        self.scale = float(scale)
        self.translation = translation.astype(float)
        self.default_translation = self.translation.copy()
        self.min_scale = self.default_scale * 0.35
        self.max_scale = self.default_scale * 8.0
        self.zoom_step = 1.2

        self._dragging = False
        self._last_mouse_pos: Optional[Tuple[int, int]] = None
        self._phase_palette = [
            (90, 220, 140),
            (250, 205, 90),
            (140, 185, 255),
            (255, 150, 190),
            (255, 180, 120),
        ]

        self.font = pygame.font.SysFont("Arial", 16)
        self.font_small = pygame.font.SysFont("Arial", 12)
        self.font_micro = pygame.font.SysFont("Arial", 10)

        self.road_polylines = self._cache_road_polylines()
        self.edge_midpoints = self._compute_edge_midpoints()

    def close(self) -> None:
        if pygame:
            pygame.quit()

    # ------------------------------------------------------------------ #
    # Event handling                                                     #
    # ------------------------------------------------------------------ #

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key in (pygame.K_r, pygame.K_SPACE):
                    self._reset_view()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self._dragging = True
                    self._last_mouse_pos = event.pos
                elif event.button == 4:
                    self._zoom_at(event.pos, self.zoom_step)
                elif event.button == 5:
                    self._zoom_at(event.pos, 1.0 / self.zoom_step)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self._dragging = False
                    self._last_mouse_pos = None
            elif event.type == pygame.MOUSEMOTION and self._dragging:
                self._pan(event.rel)
                self._last_mouse_pos = event.pos
        return True

    # ------------------------------------------------------------------ #
    # Rendering                                                          #
    # ------------------------------------------------------------------ #

    def draw(
        self,
        vehicles: Iterable[Vehicle],
        signals: Dict[int, TrafficSignal],
        incidents: Dict[EdgeKey, Incident],
        metrics: Dict[str, float],
    ) -> None:
        self.screen.fill((14, 18, 28))
        self._draw_roads(signals, incidents)
        self._draw_incident_markers(incidents)
        self._draw_signals(signals)
        self._draw_vehicles(vehicles)
        self._draw_overlay(metrics, len(incidents))
        pygame.display.flip()

    def tick(self, fps: int = 60) -> None:
        self.clock.tick(fps)

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _cache_road_polylines(self) -> Dict[EdgeKey, np.ndarray]:
        polylines: Dict[EdgeKey, np.ndarray] = {}
        for edge in self.network.iter_edges():
            coords = np.asarray(edge.geometry.coords, dtype=float)
            if coords.shape[0] < 4 and edge.geometry.length > 0:
                samples = max(4, int(edge.geometry.length / 8))
                distances = np.linspace(0.0, edge.geometry.length, samples)
                sampled_points = [
                    edge.geometry.interpolate(float(dist)) for dist in distances
                ]
                coords = np.array([(pt.x, pt.y) for pt in sampled_points], dtype=float)
            polylines[edge.key] = coords
        return polylines

    def _compute_edge_midpoints(self) -> Dict[EdgeKey, np.ndarray]:
        midpoints: Dict[EdgeKey, np.ndarray] = {}
        for edge in self.network.iter_edges():
            geometry = edge.geometry
            distance = max(0.0, min(geometry.length * 0.5, geometry.length))
            point = geometry.interpolate(distance)
            midpoints[edge.key] = np.array([point.x, point.y], dtype=float)
        return midpoints

    def _pan(self, delta: Tuple[int, int]) -> None:
        dx, dy = delta
        self.translation[0] -= float(dx)
        self.translation[1] += float(dy)

    def _zoom_at(self, mouse_pos: Tuple[int, int], factor: float) -> None:
        new_scale = self.scale * factor
        new_scale = max(self.min_scale, min(self.max_scale, new_scale))
        if abs(new_scale - self.scale) < 1e-6:
            return

        world_pos = self._screen_to_world(mouse_pos)
        self.scale = new_scale
        self.translation[0] = mouse_pos[0] - world_pos[0] * self.scale
        self.translation[1] = (self.screen_rect.height - mouse_pos[1]) - world_pos[1] * self.scale

    def _reset_view(self) -> None:
        self.scale = self.default_scale
        self.translation = self.default_translation.copy()

    def _zoom_ratio(self) -> float:
        if self.default_scale <= 0:
            return 1.0
        return max(0.3, min(3.0, self.scale / self.default_scale))

    def _ui_scale(self) -> float:
        return max(0.55, min(2.0, math.sqrt(self._zoom_ratio())))

    def _transform_points(self, points: np.ndarray, *, to_int: bool = True) -> np.ndarray:
        transformed = points * self.scale
        transformed = transformed + self.translation
        transformed[:, 1] = self.screen_rect.height - transformed[:, 1]
        if to_int:
            return transformed.astype(int)
        return transformed

    def _world_to_screen(self, coord: Tuple[float, float] | np.ndarray) -> Tuple[int, int]:
        point = np.asarray(coord, dtype=float)
        transformed = self._transform_points(point.reshape(1, 2))
        return int(transformed[0, 0]), int(transformed[0, 1])

    def _screen_to_world(self, pos: Tuple[int, int]) -> np.ndarray:
        x, y = pos
        world_x = (x - self.translation[0]) / self.scale
        world_y = ((self.screen_rect.height - y) - self.translation[1]) / self.scale
        return np.array([world_x, world_y], dtype=float)

    def _point_on_screen(self, pos: Tuple[int, int], margin: int = 32) -> bool:
        x, y = pos
        return (
            -margin <= x <= self.screen_rect.width + margin
            and -margin <= y <= self.screen_rect.height + margin
        )

    # ------------------------------------------------------------------ #
    # Drawing primitives                                                 #
    # ------------------------------------------------------------------ #

    def _draw_roads(
        self,
        signals: Dict[int, TrafficSignal],
        incidents: Dict[EdgeKey, Incident],
    ) -> None:
        for edge_key, coords in self.road_polylines.items():
            if coords.shape[0] < 2:
                continue

            screen_points = self._transform_points(coords)
            if screen_points.shape[0] < 2:
                continue

            edge = self.network.edges[edge_key]
            incident = incidents.get(edge_key)
            signal = signals.get(edge_key[1])
            highlight_green = bool(signal and signal.is_green(edge_key))

            color = self._road_color(edge, highlight_green, incident)
            width = self._road_width(edge)

            pygame.draw.lines(self.screen, color, False, screen_points.tolist(), width)

            if incident:
                overlay_color = (255, 210, 120)
                pygame.draw.lines(self.screen, overlay_color, False, screen_points.tolist(), 1)

    def _road_color(
        self,
        edge: RoadEdge,
        highlight_green: bool,
        incident: Optional[Incident],
    ) -> Tuple[int, int, int]:
        lane_factor = min(edge.lanes, 4)
        base_color = (
            50 + lane_factor * 18,
            80 + lane_factor * 12,
            130 + lane_factor * 10,
        )

        if edge.is_closed:
            return (200, 80, 80)
        if incident:
            severity = incident.severity
            return (
                int(170 + 70 * severity),
                int(70 + 40 * severity),
                int(55 + 35 * severity),
            )
        if highlight_green:
            return (90, 210, 140)
        return base_color

    def _road_width(self, edge: RoadEdge) -> int:
        zoom = self._zoom_ratio()
        base = 1.6 + edge.lanes * 0.9
        width = int(max(1.0, base * zoom ** 0.6))
        return max(1, width)

    def _draw_incident_markers(self, incidents: Dict[EdgeKey, Incident]) -> None:
        if not incidents:
            return

        ticks = pygame.time.get_ticks() if pygame else 0
        ui_scale = self._ui_scale()
        for edge_key, incident in incidents.items():
            midpoint = self.edge_midpoints.get(edge_key)
            if midpoint is None:
                continue

            pos = self._world_to_screen(midpoint)
            if not self._point_on_screen(pos):
                continue

            severity = float(incident.severity)
            base_radius = 6.0 + severity * 8.0
            pulse = 1.0 + 0.25 * math.sin(ticks / 250.0 + severity * 2.0)
            radius = int(max(4, base_radius * pulse * ui_scale))

            color = (
                min(255, int(220 + 35 * severity)),
                int(120 + 90 * severity),
                int(60 + 60 * severity),
            )
            pygame.draw.circle(self.screen, color, pos, radius, width=2)
            pygame.draw.circle(
                self.screen,
                (12, 10, 18),
                pos,
                max(2, int(radius * 0.55)),
                width=0,
            )

            label = incident.kind.value[0].upper()
            text = self.font_micro.render(label, True, (255, 240, 220))
            self.screen.blit(
                text,
                (pos[0] - text.get_width() // 2, pos[1] - text.get_height() // 2),
            )

    def _draw_signals(self, signals: Dict[int, TrafficSignal]) -> None:
        ui_scale = self._ui_scale()
        for node_id, signal in signals.items():
            node = self.network.nodes[node_id]
            pos = self._world_to_screen((node.x, node.y))
            if not self._point_on_screen(pos, margin=24):
                continue

            outer_radius = int(max(7, 10 * ui_scale))
            ring_rect = pygame.Rect(0, 0, outer_radius * 2, outer_radius * 2)
            ring_rect.center = pos

            pygame.draw.circle(self.screen, (18, 24, 36), pos, outer_radius + 3)
            pygame.draw.circle(self.screen, (28, 38, 56), pos, outer_radius + 3, width=2)

            phase_count = len(signal.phases)
            if phase_count:
                for idx, phase in enumerate(signal.phases):
                    start_angle = -math.pi / 2 + idx * (2 * math.pi / phase_count)
                    end_angle = start_angle + (2 * math.pi / phase_count)
                    color = self._phase_palette[idx % len(self._phase_palette)]
                    width = 3 if idx == signal.current_phase_index else 2
                    if idx != signal.current_phase_index:
                        color = tuple(int(c * 0.45) for c in color)
                    pygame.draw.arc(self.screen, color, ring_rect, start_angle, end_angle, width)

            inner_radius = max(4, outer_radius - 4)
            pygame.draw.circle(self.screen, (12, 18, 28), pos, inner_radius)
            pygame.draw.circle(self.screen, (45, 62, 82), pos, inner_radius, width=1)

            if phase_count:
                active_color = self._phase_palette[
                    signal.current_phase_index % len(self._phase_palette)
                ]
                progress = 0.0
                if signal.target_duration > 1e-6:
                    progress = min(1.0, signal.time_in_phase / signal.target_duration)
                progress_rect = pygame.Rect(
                    0, 0, (inner_radius - 1) * 2, (inner_radius - 1) * 2
                )
                progress_rect.center = pos
                if progress > 0.01:
                    pygame.draw.arc(
                        self.screen,
                        active_color,
                        progress_rect,
                        -math.pi / 2,
                        -math.pi / 2 + 2 * math.pi * progress,
                        width=2,
                    )

                if ui_scale >= 0.7:
                    remaining = max(0.0, signal.target_duration - signal.time_in_phase)
                    timer_text = self.font_small.render(f"{remaining:2.0f}", True, (235, 235, 235))
                    self.screen.blit(
                        timer_text,
                        (
                            pos[0] - timer_text.get_width() // 2,
                            pos[1] - timer_text.get_height() // 2,
                        ),
                    )
                if ui_scale >= 1.1:
                    label = self.font_micro.render(signal.current_phase().name, True, (220, 220, 220))
                    self.screen.blit(
                        label,
                        (pos[0] + outer_radius + 4, pos[1] - label.get_height() // 2),
                    )

    def _draw_vehicles(self, vehicles: Iterable[Vehicle]) -> None:
        zoom = self._zoom_ratio()
        for vehicle in vehicles:
            if vehicle.destination_reached():
                continue
            edge = self.network.edges[vehicle.current_edge()]
            point = _point_along(edge.geometry, vehicle.distance_on_edge)
            pos = self._world_to_screen(point)
            if not self._point_on_screen(pos):
                continue

            size = int(max(2, vehicle.profile.length_m * 0.35 * zoom))
            color = _vehicle_color(vehicle.profile.name)
            pygame.draw.circle(self.screen, color, pos, size)

    def _draw_overlay(self, metrics: Dict[str, float], incident_count: int) -> None:
        if not self.config.enable_gui_overlays:
            return

        sim_time = metrics.get("time_s", 0.0)
        minutes = int(sim_time // 60)
        seconds = int(sim_time % 60)
        fps = self.clock.get_fps()
        zoom = self._zoom_ratio()

        lines = [
            f"Sim time: {minutes:02d}:{seconds:02d}",
            f"Vehicles: {int(metrics.get('active_vehicles', 0))}",
            f"Completed: {int(metrics.get('completed_trips', 0))}",
            f"Queue: {metrics.get('total_queue', 0.0):.0f} veh",
            f"Avg speed: {metrics.get('avg_speed_mps', 0.0):.1f} m/s",
            f"Avg fuel: {metrics.get('avg_fuel_l_per_trip', 0.0):.2f} L",
            f"Incidents: {incident_count}",
            f"FPS: {fps:4.0f} | Zoom: {zoom:.2f}x",
        ]

        y = 12
        for text in lines:
            label = self.font.render(text, True, (235, 235, 235))
            self.screen.blit(label, (12, y))
            y += 18

        instructions = [
            "Mouse drag: pan",
            "Scroll: zoom",
            "R / Space: reset view",
        ]

        x = self.screen_rect.width - 190
        y = 12
        for text in instructions:
            label = self.font_small.render(text, True, (210, 210, 210))
            self.screen.blit(label, (x, y))
            y += 16


def _point_along(geometry: LineString, distance: float) -> np.ndarray:
    if geometry.length <= 0:
        x, y = geometry.coords[0]
        return np.array([x, y], dtype=float)
    clipped_distance = max(0.0, min(distance, geometry.length))
    point = geometry.interpolate(clipped_distance)
    return np.array([point.x, point.y], dtype=float)


def _vehicle_color(name: str) -> Tuple[int, int, int]:
    palette = {
        "sedan": (90, 190, 255),
        "suv": (255, 150, 110),
        "delivery_van": (255, 225, 130),
        "semi": (215, 140, 255),
    }
    return palette.get(name, (200, 200, 200))
