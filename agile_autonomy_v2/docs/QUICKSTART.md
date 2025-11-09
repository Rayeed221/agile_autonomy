# Quick Start Guide

## Installation

### 1. Clone the Repository

```bash
cd agile_autonomy_v2
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n agile_autonomy python=3.9
conda activate agile_autonomy
```

### 3. Install Package

```bash
pip install -e .
```

This will install the package in development mode with all dependencies.

## Running Your First Simulation

### Test Installation

```bash
python scripts/test_basic.py
```

This runs basic tests to verify everything is working.

### Run Demo

```bash
python scripts/demo_simulation.py
```

This opens an interactive menu with several demos:
1. **Hover** - Simple hovering demonstration
2. **Waypoint Navigation** - Fly to predefined waypoints
3. **Trajectory Following** - Follow a circular trajectory
4. **Sensors** - Read and display sensor data

## Basic Usage

### Create a Simulator

```python
from agile_autonomy.simulation import QuadrotorSimulator
from agile_autonomy.core import Config

# Load configuration
config = Config.from_yaml("config/sim_config.yaml")

# Create simulator
sim = QuadrotorSimulator(config=config, gui=True)

# Reset to starting position
state = sim.reset(position=[0, 0, 2.0])

# Run simulation loop
for i in range(1000):
    # Hover at current position
    state = sim.step()

    # Or fly to a target
    # state = sim.step(target_position=[5, 0, 2])

    # Render
    sim.render()

sim.close()
```

### Read Sensors

```python
# Get all sensor data
sensors = sim.get_sensor_data()

rgb_image = sensors['rgb']        # (480, 640, 3) RGB image
depth_image = sensors['depth']    # (480, 640) depth in meters
accel = sensors['imu_accel']      # (3,) acceleration
gyro = sensors['imu_gyro']        # (3,) angular velocity
```

### Execute a Trajectory

```python
import numpy as np
from agile_autonomy.core import Trajectory

# Create trajectory
positions = np.array([
    [0, 0, 2],
    [5, 0, 2],
    [5, 5, 2],
    [0, 5, 2],
])

trajectory = Trajectory.from_positions(positions, dt=0.1)

# Execute
result = sim.run_trajectory(trajectory, max_time=30.0)

print(f"Success: {result['success']}")
print(f"Collisions: {sum(result['collisions'])}")
```

## Configuration

All configurations are in `config/` directory as YAML files.

### Edit Simulation Config

```yaml
# config/sim_config.yaml

environment:
  type: "forest"
  tree_spacing: 5.0      # Adjust for denser/sparser forest
  num_trees: 100

controller:
  position_p: 1.0        # Increase for faster convergence
  position_d: 0.5        # Increase for more damping
```

Load custom config:

```python
config = Config.from_yaml("my_config.yaml")
sim = QuadrotorSimulator(config=config)
```

## Next Steps

1. **Collect data** - See `docs/data_collection.md`
2. **Train a model** - See `docs/training.md`
3. **Deploy model** - See `docs/deployment.md`
4. **Understand architecture** - See `PROJECT_ANALYSIS.md`

## Troubleshooting

### PyBullet GUI not showing

Make sure you have OpenGL support. On Linux:
```bash
sudo apt-get install libgl1-mesa-glx
```

### ImportError

Make sure you installed in development mode:
```bash
pip install -e .
```

### Camera rendering issues

PyBullet requires OpenGL. If running remotely, you may need to use `gui=False` or set up X forwarding.

## Getting Help

- Check the full documentation in `docs/`
- See `PROJECT_ANALYSIS.md` for system architecture
- Open an issue on GitHub
