from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import networkx as nx
import osmnx as ox

from .config import SimulationConfig

LOGGER = logging.getLogger(__name__)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_graph(config: SimulationConfig, force_refresh: bool = False) -> nx.MultiDiGraph:
    """Load a road network graph, caching the result if requested."""

    cache_path = config.graph_cache
    graph: Optional[nx.MultiDiGraph] = None

    if cache_path.exists() and not force_refresh:
        LOGGER.info("Loading graph from cache at %s", cache_path)
        graph = ox.load_graphml(cache_path)
    else:
        LOGGER.info("Fetching graph for %s", config.place)
        graph = fetch_graph(
            place=config.place,
            distance=config.distance,
            bbox=config.lat_lon_bbox,
        )
        _ensure_parent(cache_path)
        LOGGER.info("Saving graph cache to %s", cache_path)
        ox.save_graphml(graph, cache_path)

    return ox.project_graph(graph)


def fetch_graph(
    place: Optional[str] = None,
    distance: Optional[int] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> nx.MultiDiGraph:
    """Download a drivable road network from OpenStreetMap."""

    if bbox:
        north, south, east, west = bbox
        graph = ox.graph_from_bbox(
            north=north,
            south=south,
            east=east,
            west=west,
            network_type="drive",
            simplify=True,
        )
    elif place:
        kwargs = {
            "query": place,
            "network_type": "drive",
            "simplify": True,
        }
        if distance is not None:
            # osmnx >= 2.0 renamed the distance parameter
            kwargs["buffer_dist"] = distance
        try:
            graph = ox.graph_from_place(**kwargs)
        except TypeError as exc:
            LOGGER.warning("graph_from_place failed for %s (%s); falling back to point buffer", place, exc)
            if distance is None:
                distance = 1500
            center_point = ox.geocode(place)
            graph = ox.graph_from_point(
                center_point,
                dist=float(distance),
                network_type="drive",
                simplify=True,
            )
    else:
        raise ValueError("place or bbox must be provided to fetch a graph")

    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    return graph
