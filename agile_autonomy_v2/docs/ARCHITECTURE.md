# Architecture Overview

## System Components

The Agile Autonomy v2 system consists of several modular components:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  (Training, Testing, Data Collection Scripts)               │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────┴────────────────────────────────────────────┐
│                    Core Modules                              │
├──────────────────────────────────────────────────────────────┤
│  Models       │  Data       │  Planning    │  Utils          │
│  - PlaNet     │  - Datasets │  - MPPI      │  - Geometry    │
│  - Losses     │  - Loaders  │  - Reference │  - Transforms  │
│               │             │              │  - Viz         │
└────────────────┴──────┬──────┴──────────────┴────────────────┘
                        │
┌───────────────────────┴──────────────────────────────────────┐
│                  Simulation Layer                             │
├──────────────────────────────────────────────────────────────┤
│  Simulator  │  Environment │  Sensors     │  Controller      │
│  - Main     │  - Forest    │  - Camera    │  - PID          │
│  - Quadrotor│  - Obstacles │  - Depth     │  - MPC          │
│             │              │  - IMU       │                  │
└──────────────┴──────────────┴──────────────┴──────────────────┘
                        │
┌───────────────────────┴──────────────────────────────────────┐
│                   PyBullet Physics                            │
│  - Rigid body dynamics                                       │
│  - Collision detection                                       │
│  - Rendering (OpenGL)                                        │
└──────────────────────────────────────────────────────────────┘
```

## Core Data Structures

### QuadrotorState

Represents the complete state of the quadrotor:

```python
@dataclass
class QuadrotorState:
    position: np.ndarray          # [x, y, z] in world frame
    velocity: np.ndarray          # [vx, vy, vz] in world frame
    orientation: Rotation         # scipy Rotation object
    angular_velocity: np.ndarray  # [wx, wy, wz] in body frame
    timestamp: float              # Time in seconds
```

Key methods:
- `rotation_matrix`: Get 3x3 rotation matrix
- `quaternion`: Get quaternion [w, x, y, z]
- `euler_angles`: Get roll, pitch, yaw
- `velocity_body_frame`: Velocity in body frame
- `to_network_input()`: Convert to neural network input format

### Trajectory

Represents a sequence of waypoints:

```python
class Trajectory:
    points: List[TrajectoryPoint]  # Waypoint sequence
    frame: str                     # 'world' or 'body'
    dt: float                      # Time step between points
```

Key methods:
- `from_positions()`: Create from position array
- `from_network_output()`: Create from neural network output
- `transform_to_world_frame()`: Convert body→world
- `get_point_at_time()`: Interpolate at specific time
- `compute_velocities()`: Calculate velocities from positions

### Config

Hierarchical configuration management using OmegaConf:

```python
config = Config.from_yaml("config.yaml")
value = config.get("model.backbone")
config.set("training.batch_size", 16)
```

## Simulation Components

### QuadrotorSimulator

Main simulator class that integrates all components:

**Responsibilities:**
- Physics simulation (PyBullet)
- Environment management
- Sensor simulation
- Control loop execution

**Key Methods:**
```python
sim = QuadrotorSimulator(config, gui=True)
state = sim.reset(position=[0, 0, 2])
state = sim.step(target_position=[5, 0, 2])
sensors = sim.get_sensor_data()
result = sim.run_trajectory(trajectory)
```

### Quadrotor

PyBullet-based quadrotor model:

**Features:**
- Rigid body dynamics
- Thrust and torque control
- Collision detection
- Visual representation with rotors

**Physics:**
- Mass: 0.775 kg (configurable)
- 4 rotors (visualization only)
- Thrust applied in body +Z direction
- Torques in body frame

### Environment

Procedural environment generation:

**ForestEnvironment:**
- Generates random tree positions
- Configurable spacing and density
- Collision-enabled cylinders
- Point cloud extraction

**ObstacleEnvironment:**
- Random boxes and cylinders
- Variable sizes and positions
- Denser obstacle fields

### Sensors

**DepthCamera:**
- RGB image capture (640×480 default)
- Depth image (meters)
- Configurable FOV, resolution
- Mounted on quadrotor body

**IMU:**
- 3-axis accelerometer
- 3-axis gyroscope
- Optional noise simulation
- Body frame measurements

### Controller

**PIDController:**
- Cascaded position + attitude control
- Outer loop: Position PID → desired thrust + attitude
- Inner loop: Attitude PID → torques
- Configurable gains and limits

**Control Flow:**
```
Target Position → Position Error → Desired Acceleration
                                          ↓
                              Thrust + Desired Attitude
                                          ↓
                          Attitude Error → Torques
                                          ↓
                          Quadrotor Motors
```

## Data Flow

### Training Pipeline

```
1. Data Collection:
   Simulator → Expert Planner → Record (states, images, trajectories)

2. Dataset Creation:
   Rollouts → DataLoader → Batches

3. Training:
   Batch → Network → Predictions
        → Loss (position + collision cost)
        → Backprop

4. Checkpoint:
   Save model weights every N epochs
```

### Inference Pipeline

```
1. Sensor Input:
   Depth Camera → Image (224×224)
   IMU → State vector

2. Network:
   [Image, State] → PlaNet → 3 Trajectory Modes

3. Selection:
   Evaluate collision cost → Pick best mode

4. Execution:
   Trajectory → Controller → Motor commands
```

## Module Details

### agile_autonomy/core/

**state.py**
- QuadrotorState dataclass
- Coordinate transformations
- Serialization

**trajectory.py**
- Trajectory and TrajectoryPoint classes
- Interpolation
- Frame transformations

**config.py**
- Config management with OmegaConf
- YAML loading/saving
- Hierarchical access

### agile_autonomy/simulation/

**quadrotor.py**
- Quadrotor model in PyBullet
- Physics integration
- Collision queries

**environment.py**
- Environment base class
- ForestEnvironment
- ObstacleEnvironment
- Point cloud generation

**sensors.py**
- Camera (RGB)
- DepthCamera (RGB-D)
- IMU (accel + gyro)

**controller.py**
- PIDController
- SimpleHoverController
- Trajectory tracking

**simulator.py**
- QuadrotorSimulator (main class)
- Integration of all components
- Simulation loop

### agile_autonomy/models/

(To be implemented)
- PlaNet neural network
- Backbone networks (MobileNet)
- Loss functions

### agile_autonomy/data/

(To be implemented)
- Dataset classes
- Data loaders
- Transforms and augmentation

### agile_autonomy/planning/

(To be implemented)
- MPPI planner (expert)
- Reference trajectory generation
- Collision checking

### agile_autonomy/utils/

(To be implemented)
- Geometry utilities
- Visualization tools
- Metrics computation

## Design Principles

1. **Modularity**: Each component is self-contained and testable
2. **No ROS**: Pure Python, no ROS dependencies
3. **Type hints**: All functions have type annotations
4. **Configuration**: YAML-based, hierarchical configs
5. **Extensibility**: Easy to add new environments, sensors, controllers
6. **Testability**: Unit tests for each module

## Extension Points

### Adding a New Environment

```python
class MyEnvironment(Environment):
    def __init__(self, client_id: int, **kwargs):
        super().__init__(client_id)
        # Your init

    def generate(self) -> None:
        # Create your environment objects
        pass
```

### Adding a New Sensor

```python
class LiDAR:
    def __init__(self, client_id: int, num_rays: int = 64):
        self.client_id = client_id
        self.num_rays = num_rays

    def scan(self, quad_state: QuadrotorState) -> np.ndarray:
        # Implement ray casting
        pass
```

### Adding a New Controller

```python
class MPCController:
    def compute_control(
        self,
        state: QuadrotorState,
        trajectory: Trajectory
    ) -> Tuple[float, np.ndarray]:
        # Implement MPC optimization
        pass
```

## Performance Considerations

- **Physics timestep**: 240 Hz (default) for stability
- **Control frequency**: 50-100 Hz (typical)
- **Network inference**: 15 Hz (target)
- **Rendering**: Decoupled from physics (optional)

## Future Enhancements

- [ ] GPU-accelerated physics (Isaac Gym)
- [ ] Parallel environments for training
- [ ] Real-time factor >1.0
- [ ] Hardware deployment (ROS2 bridge)
- [ ] Multi-agent simulation
