# Agile Autonomy v2 - Pure Python Implementation

Modern Python-only implementation of the Agile Autonomy project for high-speed quadrotor navigation in cluttered environments.

## Features

- **No ROS dependencies** - Pure Python implementation
- **Modern ML stack** - PyTorch 2.1+ (or TensorFlow 2.15+)
- **Easy simulation** - PyBullet-based physics and rendering
- **Clean architecture** - Modular, testable, well-documented
- **Simple setup** - `pip install -e .` and you're ready

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run a simple simulation demo
python scripts/demo_simulation.py

# Train the network
python scripts/train.py --config config/train_config.yaml

# Test trained model
python scripts/test.py --config config/test_config.yaml
```

## Project Structure

```
agile_autonomy_v2/
├── agile_autonomy/          # Main package
│   ├── models/              # Neural network architectures
│   ├── data/                # Data loading and processing
│   ├── simulation/          # PyBullet simulator
│   ├── planning/            # Trajectory planning (MPPI)
│   ├── utils/               # Utilities (geometry, transforms, viz)
│   └── core/                # Core abstractions
├── scripts/                 # Training and testing scripts
├── config/                  # Configuration files
├── notebooks/               # Jupyter notebooks for analysis
└── tests/                   # Unit tests
```

## Installation

### Prerequisites

- Python 3.8+
- (Optional) CUDA-capable GPU for training

### Install

```bash
# Clone the repository
cd agile_autonomy_v2

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

## Usage

### Running Simulation

```python
from agile_autonomy.simulation import QuadrotorSimulator
from agile_autonomy.simulation.environment import ForestEnvironment

# Create environment
env = ForestEnvironment(tree_spacing=5.0, area_size=(100, 100))

# Create simulator
sim = QuadrotorSimulator(environment=env)

# Run simulation
for _ in range(1000):
    state = sim.get_state()
    action = controller.compute_action(state)
    sim.step(action)
    sim.render()
```

### Training

```bash
# Train from scratch
python scripts/train.py --config config/train_config.yaml

# Fine-tune existing checkpoint
python scripts/train.py --config config/train_config.yaml --checkpoint data/checkpoints/planet_ckpt.pt
```

### Data Collection

```bash
# Collect training data with expert
python scripts/collect_data.py --config config/data_collection.yaml --num_rollouts 100
```

## Architecture

### Neural Network (PlaNet)

- **Input**: Depth images (224×224) + IMU data (position, attitude, velocity, body rates)
- **Output**: 3 trajectory modes, each with 10 waypoints (1 second prediction)
- **Backbone**: MobileNetV2 (pretrained on ImageNet)
- **Frequency**: 15 Hz real-time inference

### Simulator

- **Physics**: PyBullet (240 Hz)
- **Rendering**: OpenGL via PyBullet
- **Sensors**: RGB camera, depth camera, IMU
- **Environments**: Procedural forests, obstacle fields

### Controller

- **Low-level**: PID controller for position/attitude
- **High-level**: Neural network trajectory planning
- **Expert**: MPPI (Model Predictive Path Integral)

## Configuration

All configurations use YAML files in the `config/` directory:

- `train_config.yaml` - Training parameters
- `test_config.yaml` - Testing/evaluation parameters
- `sim_config.yaml` - Simulator settings
- `data_collection.yaml` - Data collection settings

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black agile_autonomy/ scripts/ tests/

# Lint
ruff check agile_autonomy/ scripts/ tests/
```

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{Loquercio2021Science,
  title={Learning High-Speed Flight in the Wild},
  author={Loquercio, Antonio and Kaufmann, Elia and Ranftl, Ren{\'e} and M{\"u}ller, Matthias and Koltun, Vladlen and Scaramuzza, Davide},
  journal={Science Robotics},
  year={2021},
  month={October}
}
```

## License

Same as original project.

## Acknowledgments

This is a modernized implementation of the original [agile_autonomy](https://github.com/uzh-rpg/agile_autonomy) project by UZH-RPG, rebuilt with modern Python tools and without ROS dependencies.

Original authors: Antonio Loquercio, Elia Kaufmann, and team at UZH Robotics and Perception Group.
