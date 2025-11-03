from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .config import SimulationConfig
from .simulation import Simulation
from .train import app as train_app
from .visualization import PygameVisualizer


cli = typer.Typer(help="Traffic simulation and training toolkit.")
cli.add_typer(train_app, name="train")


@cli.command("run")
def run_simulation(
    place: str = typer.Option("Downtown Los Angeles, California, USA", help="Location to download from OSM."),
    minutes: float = typer.Option(30.0, help="Duration of the simulation in minutes."),
    dt: float = typer.Option(0.5, help="Simulation step size in seconds."),
    spawn_rate: float = typer.Option(120.0, help="Vehicle spawn rate (vehicles per minute)."),
    max_vehicles: int = typer.Option(250, help="Maximum concurrent vehicles."),
    incident_rate: float = typer.Option(4.0, help="Incidents per simulated hour."),
    seed: int = typer.Option(42, help="Random seed for reproducibility."),
    headless: bool = typer.Option(False, help="Run without a pygame window."),
    screen_width: int = typer.Option(1280, help="Pygame viewport width."),
    screen_height: int = typer.Option(720, help="Pygame viewport height."),
    refresh_graph: bool = typer.Option(False, help="Force re-download of the OSM graph."),
    graph_cache: Optional[Path] = typer.Option(Path("data/osm_graph.graphml"), help="Cache file for the downloaded graph."),
) -> None:
    """Run a single simulation episode with optional visualisation."""

    config = SimulationConfig(
        place=place,
        simulation_minutes=minutes,
        dt=dt,
        spawn_rate_per_minute=spawn_rate,
        max_vehicles=max_vehicles,
        incident_rate_per_hour=incident_rate,
        random_seed=seed,
        headless=headless,
        screen_size=(screen_width, screen_height),
        graph_cache=graph_cache or Path("data/osm_graph.graphml"),
    )

    sim = Simulation(config, force_refresh_graph=refresh_graph)
    visualizer: Optional[PygameVisualizer] = None
    if not headless:
        try:
            visualizer = PygameVisualizer(sim.network, config)
            sim.visualizer = visualizer
        except ImportError as exc:
            typer.echo(f"pygame unavailable ({exc}); running headless instead.", err=True)
            sim.config.headless = True
            sim.visualizer = None
        except Exception as exc:  # pragma: no cover - graphical init edge cases
            typer.echo(f"Failed to initialise pygame visualiser: {exc}", err=True)
            sim.config.headless = True
            sim.visualizer = None

    report = sim.run()
    typer.echo(
        f"Completed {report.time_s/60:.1f} simulated minutes. "
        f"Trips: {report.completed_trips}, "
        f"avg speed {report.avg_speed_mps:.2f} m/s, "
        f"avg fuel {report.avg_fuel_l_per_trip:.2f} L."
    )

    if visualizer:
        visualizer.close()


def main() -> None:
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
