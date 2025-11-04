from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class VehicleProfile:
    """Parameterized vehicle model used by the simulator."""

    name: str
    max_speed_mps: float
    acceleration_mps2: float
    length_m: float
    base_fuel_consumption_l_per_100km: float
    fuel_penalty_per_mps2: float


DEFAULT_VEHICLE_PROFILES: Dict[str, VehicleProfile] = {
    "sedan": VehicleProfile(
        name="sedan",
        max_speed_mps=30.0,
        acceleration_mps2=2.5,
        length_m=4.5,
        base_fuel_consumption_l_per_100km=7.5,
        fuel_penalty_per_mps2=0.6,
    ),
    "suv": VehicleProfile(
        name="suv",
        max_speed_mps=28.0,
        acceleration_mps2=2.1,
        length_m=5.0,
        base_fuel_consumption_l_per_100km=9.0,
        fuel_penalty_per_mps2=0.8,
    ),
    "delivery_van": VehicleProfile(
        name="delivery_van",
        max_speed_mps=25.0,
        acceleration_mps2=1.7,
        length_m=5.8,
        base_fuel_consumption_l_per_100km=11.0,
        fuel_penalty_per_mps2=1.0,
    ),
    "semi": VehicleProfile(
        name="semi",
        max_speed_mps=22.0,
        acceleration_mps2=1.0,
        length_m=12.0,
        base_fuel_consumption_l_per_100km=28.0,
        fuel_penalty_per_mps2=2.5,
    ),
}


@dataclass
class SimulationConfig:
    """Runtime configuration for the traffic simulator."""

    place: str = "Downtown Los Angeles, California, USA"
    distance: int | None = None
    lat_lon_bbox: Tuple[float, float, float, float] | None = None
    graph_cache: Path = Path("data/osm_graph.graphml")
    dt: float = 0.5
    simulation_minutes: float = 45.0
    headless: bool = False
    screen_size: Tuple[int, int] = (1280, 720)
    spawn_rate_per_minute: float = 120.0
    max_vehicles: int = 250
    random_seed: int = 42
    incident_rate_per_hour: float = 4.0
    vehicle_mix: Dict[str, float] = field(
        default_factory=lambda: {
            "sedan": 0.55,
            "suv": 0.2,
            "delivery_van": 0.15,
            "semi": 0.1,
        }
    )
    enable_gui_overlays: bool = True
    export_metrics: bool = False
    metrics_output: Path = Path("output/metrics.csv")
    enable_vsync: bool = True
    use_scaled_display: bool = True
    hardware_acceleration: bool = True
    antialias_rendering: bool = True
    signal_warmup_seconds: float = 120.0
    signal_yellow_duration: float = 3.0

    def total_steps(self) -> int:
        return int(self.simulation_minutes * 60 / self.dt)

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.random_seed)

    def normalized_vehicle_mix(self) -> Dict[str, float]:
        mix = np.array(list(self.vehicle_mix.values()), dtype=float)
        if mix.sum() <= 0:
            raise ValueError("Vehicle mix weights must sum to a positive value")
        mix = mix / mix.sum()
        return dict(zip(self.vehicle_mix.keys(), mix))
