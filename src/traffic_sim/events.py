from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .road_network import EdgeKey, RoadNetwork


class IncidentType(str, enum.Enum):
    ACCIDENT = "accident"
    ROADWORK = "roadwork"
    WEATHER = "weather"


@dataclass
class Incident:
    incident_id: int
    edge_key: EdgeKey
    kind: IncidentType
    start_time: float
    duration: float
    severity: float
    cleared: bool = False

    def end_time(self) -> float:
        return self.start_time + self.duration


class IncidentManager:
    """Simple stochastic incident generator."""

    def __init__(self, network: RoadNetwork, hourly_rate: float, rng: np.random.Generator):
        self.network = network
        self.hourly_rate = hourly_rate
        self.rng = rng
        self.active_incidents: Dict[int, Incident] = {}
        self._next_id = 1

    def update(self, current_time: float, dt: float) -> List[Incident]:
        resolved: List[Incident] = []

        for incident in list(self.active_incidents.values()):
            if current_time >= incident.end_time() and not incident.cleared:
                self.network.open_edge(incident.edge_key)
                incident.cleared = True
                resolved.append(incident)
                del self.active_incidents[incident.incident_id]

        expected_incidents = self.hourly_rate * dt / 3600.0
        if self.rng.random() < expected_incidents:
            incident = self._spawn_incident(current_time)
            if incident:
                self.active_incidents[incident.incident_id] = incident

        return resolved

    def _spawn_incident(self, current_time: float) -> Optional[Incident]:
        available_edges = [edge for edge in self.network.iter_edges() if not edge.is_closed]
        if not available_edges:
            return None

        edge = self.rng.choice(available_edges)
        severity = float(self.rng.uniform(0.3, 1.0))
        duration = float(self.rng.uniform(5 * 60, 20 * 60)) * severity
        kind_index = int(self.rng.integers(0, len(IncidentType)))
        kind = list(IncidentType)[kind_index]

        incident = Incident(
            incident_id=self._next_id,
            edge_key=edge.key,
            kind=kind,
            start_time=current_time,
            duration=duration,
            severity=severity,
        )

        if severity > 0.75:
            penalty = math.inf
        else:
            penalty = 1.0 + severity * 4.0

        self.network.close_edge(edge.key, penalty_multiplier=penalty)
        self._next_id += 1
        return incident
