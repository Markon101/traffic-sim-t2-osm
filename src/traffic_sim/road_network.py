from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
from shapely.geometry import LineString
from shapely.ops import substring


EdgeKey = Tuple[int, int, int]


@dataclass
class RoadNode:
    osmid: int
    x: float
    y: float
    street_count: int
    signal: bool = False


@dataclass
class RoadEdge:
    key: EdgeKey
    name: str | None
    length_m: float
    speed_mps: float
    travel_time_s: float
    base_travel_time_s: float
    lanes: int
    geometry: LineString
    oneway: bool
    attrs: Dict[str, float] = field(default_factory=dict)
    is_closed: bool = False

    def capacity(self) -> float:
        base_capacity = 1800 * self.lanes  # vehicles per hour
        if self.speed_mps < 6:
            return base_capacity * 0.4
        if self.speed_mps < 12:
            return base_capacity * 0.7
        return base_capacity


class RoadNetwork:
    """Thin wrapper around the NetworkX graph for faster lookups."""

    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self.nodes: Dict[int, RoadNode] = {}
        self.edges: Dict[EdgeKey, RoadEdge] = {}

        self._populate_nodes()
        self._populate_edges()
        self._compute_bounds()

    def _populate_nodes(self) -> None:
        for node_id, data in self.graph.nodes(data=True):
            self.nodes[node_id] = RoadNode(
                osmid=node_id,
                x=data["x"],
                y=data["y"],
                street_count=data.get("street_count", 1),
                signal=bool(data.get("highway") == "traffic_signals"),
            )

    def _populate_edges(self) -> None:
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            length = float(data.get("length", 0.0))
            speed = float(data.get("speed_kph", 30.0)) / 3.6
            travel_time = float(data.get("travel_time", length / max(speed, 0.1)))
            lanes_attr = data.get("lanes")

            if isinstance(lanes_attr, list):
                lanes = int(np.mean([float(x) for x in lanes_attr]))
            elif lanes_attr is None:
                lanes = 1
            else:
                try:
                    lanes = int(lanes_attr)
                except (ValueError, TypeError):
                    lanes = 1

            geometry = data.get("geometry")
            if geometry is None:
                geometry = LineString([(self.graph.nodes[u]["x"], self.graph.nodes[u]["y"]),
                                       (self.graph.nodes[v]["x"], self.graph.nodes[v]["y"])])

            self.edges[(u, v, k)] = RoadEdge(
                key=(u, v, k),
                name=data.get("name"),
                length_m=length,
                speed_mps=speed,
                travel_time_s=travel_time,
                base_travel_time_s=travel_time,
                lanes=max(1, lanes),
                geometry=geometry,
                oneway=bool(data.get("oneway", True)),
                attrs={
                    "grade": float(data.get("grade", 0.0)),
                    "bearing": float(data.get("bearing", 0.0)),
                },
            )

    def _compute_bounds(self) -> None:
        xs = [node.x for node in self.nodes.values()]
        ys = [node.y for node in self.nodes.values()]
        self.min_x, self.max_x = min(xs), max(xs)
        self.min_y, self.max_y = min(ys), max(ys)

    def bounds(self) -> Tuple[float, float, float, float]:
        return self.min_x, self.min_y, self.max_x, self.max_y

    def incoming_edges(self, node_id: int) -> List[RoadEdge]:
        return [self.edges[(u, v, k)] for u, v, k in self.graph.in_edges(node_id, keys=True)]

    def outgoing_edges(self, node_id: int) -> List[RoadEdge]:
        return [self.edges[(u, v, k)] for u, v, k in self.graph.out_edges(node_id, keys=True)]

    def iter_edges(self) -> Iterable[RoadEdge]:
        return self.edges.values()

    def shortest_path(self, origin: int, destination: int, weight: str = "travel_time") -> List[int]:
        return nx.shortest_path(self.graph, origin, destination, weight=weight)

    def edge_trajectory(self, edge: RoadEdge, samples: int = 50) -> np.ndarray:
        """Return sample points along an edge for visualisation."""

        if edge.length_m <= 0 or samples <= 2:
            coords = np.array(edge.geometry.coords)
            return coords[:, 0:2]

        distances = np.linspace(0, edge.length_m, samples)
        points = [substring(edge.geometry, dist, dist) for dist in distances]
        return np.array([(pt.x, pt.y) for pt in points])

    def node_coordinates(self, node_id: int) -> Tuple[float, float]:
        node = self.nodes[node_id]
        return node.x, node.y

    def is_edge_closed(self, edge_key: EdgeKey) -> bool:
        return self.edges[edge_key].is_closed

    def close_edge(self, edge_key: EdgeKey, penalty_multiplier: float | None = None) -> None:
        edge = self.edges.get(edge_key)
        if edge is None:
            return
        if penalty_multiplier is None or math.isinf(penalty_multiplier):
            edge.travel_time_s = math.inf
        else:
            edge.travel_time_s = edge.base_travel_time_s * max(1.0, penalty_multiplier)
        edge.is_closed = math.isinf(edge.travel_time_s)
        self.graph[edge_key[0]][edge_key[1]][edge_key[2]]["travel_time"] = edge.travel_time_s

    def open_edge(self, edge_key: EdgeKey) -> None:
        edge = self.edges.get(edge_key)
        if edge is None:
            return
        edge.is_closed = False
        edge.travel_time_s = edge.base_travel_time_s
        self.graph[edge_key[0]][edge_key[1]][edge_key[2]]["travel_time"] = edge.travel_time_s

    def travel_time(self, edge_key: EdgeKey) -> float:
        edge = self.edges[edge_key]
        if edge.is_closed:
            return math.inf
        return edge.travel_time_s

    def length(self, edge_key: EdgeKey) -> float:
        return self.edges[edge_key].length_m
