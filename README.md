# Traffic Simulator with RL-Controlled Signals

An adaptive traffic simulation stack that ingests real-world OpenStreetMap data, renders a rich pygame visualisation, and provides a PyTorch-based reinforcement learning harness for optimising network-wide traffic light timing. The package targets Python 3.12+ and defaults to Downtown Los Angeles (DTLA) for reproducible experimentation.

## Highlights
- **Real network geometry**: Downloads and caches drivable OSM graphs, including multi-lane roads, travel times, and signalised intersections.
- **Heterogeneous vehicles**: Multiple vehicle classes (sedans, SUVs, vans, semi-trailers) with realistic acceleration, speed caps, and fuel-use modelling.
- **Dynamic scenarios**: Stochastic spawning, car-following dynamics, adaptive signal heuristics, and incident generation (accidents, weather, roadworks) that degrade throughput or close roads.
- **Headless + GUI**: Optimised headless mode for RL training and a polished, vsync-aware pygame UI with interactive camera controls, anti-aliased roads, incident markers, and phase-progress signal rings.
- **RL environment**: Gymnasium-compatible environment exposing padded observations over the entire network and a discrete action space that selects phase/duration pairs for any signal.
- **Transformer DQN**: ~1M parameter Torch models (transformer encoder and residual MLP baseline) with CUDA-first training loop, replay buffer, and soft target updates.
- **Metrics export**: Optional CSV logging of global metrics each simulation tick.

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

> **Note**  
> PyTorch with CUDA is listed as a dependency. Install the wheel that matches your GPU/driver stack (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu121`). The project will automatically fall back to CPU if CUDA is unavailable.

## Quickstart

### Run an interactive simulation
```bash
traffic-sim run --minutes 20 --spawn-rate 150 --incident-rate 3 --no-headless
```
Key CLI options:
- `--place`: OSM place string or bounding box.
- `--refresh-graph`: Force a new map download (cached under `data/osm_graph.graphml`).
- `--screen-width/--screen-height`: Custom viewport size.
- `--headless` / `--no-headless`: Toggle pygame.

### GUI Controls
- **Mouse drag**: pan around the network.
- **Mouse wheel**: zoom in and out with focus preserved under the cursor.
- **R / Space**: reset the camera to the default framing.
- Adaptive signal rings show active phases, remaining-green countdowns, and incident markers pulse at mid-block locations for easy spotting.

> **Display tuning**  
> The `SimulationConfig` exposes `enable_vsync`, `use_scaled_display`, `hardware_acceleration`, and `antialias_rendering` flags so you can dial in visual fidelity (and disable them when profiling headless workloads).

### Train a Transformer DQN controller
```bash
traffic-sim train dqn \
  --episodes 150 \
  --batch-size 256 \
  --model-type transformer \
  --model-width 512 \
  --model-layers 8 \
  --updates-per-step 4 \
  --control-interval 6
```
Important flags:
- `--model-type`: `transformer` (default) or `residual`.
- `--model-width/--model-layers/--model-heads`: Scale the transformer capacity (and VRAM footprint).
- `--batch-size`, `--train-interval`, `--updates-per-step`: Shape GPU throughput by deciding how often and how hard each optimisation pass hits.
- `--epsilon-*`: Configure epsilon-greedy exploration schedule.
- `--control-interval`: Seconds between agent actions in the simulator.
- `--output-dir`: Checkpoint directory (defaults to `output/models`).

Checkpoints are saved as `output/models/<model>-dqn.pt` and contain model + optimiser state for continued training.

## Code Structure
- `traffic_sim/config.py`: Centralised simulation configuration and vehicle profiles.
- `traffic_sim/osm_loader.py`: Graph downloads, caching, and projection via `osmnx`.
- `traffic_sim/road_network.py`: Lightweight adapters for nodes, edges, and closures.
- `traffic_sim/simulation.py`: Main simulation loop with spawning, signalling, incidents, and metrics.
- `traffic_sim/visualization.py`: Pygame renderer with stylised roads, vehicles, and overlays.
- `traffic_sim/env.py`: Gymnasium environment for RL agents (global observation/action space).
- `traffic_sim/model.py`: Transformer and residual DQN heads.
- `traffic_sim/agent.py`: Replay buffer + DQN agent with soft target updates.
- `traffic_sim/train.py`: Typer-based training CLI (`traffic-sim train dqn`).
- `traffic_sim/cli.py`: Top-level CLI (`traffic-sim run`, `traffic-sim train ...`).

## Extending the Simulator
- **Vehicle behaviour**: Add new profiles in `config.py` or enrich `vehicle.py` with lane-changing logic.
- **Incidents**: Extend `IncidentManager` to include scheduled closures or severity-dependent detours.
- **Controllers**: Plug in custom heuristics or learned policies via `controllers.py` and `Simulation`.
- **Observations**: Modify `_signal_features` in `env.py` to expose additional metrics for learning.

---
Happy experimenting—feel free to tweak the map, spawn rates, controller logic, and reward function to explore novel traffic optimisation strategies.
