# Agile Autonomy - Complete Project Analysis & Modernization Plan

**Date:** 2025-11-09
**Original Project:** Learning High-Speed Flight in the Wild (UZH-RPG, Science Robotics 2021)
**Purpose:** Analysis for Python-only rebuild without ROS dependencies

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current Architecture Overview](#current-architecture-overview)
3. [Current Dependencies Analysis](#current-dependencies-analysis)
4. [Training Pipeline Deep Dive](#training-pipeline-deep-dive)
5. [Deployment Pipeline Deep Dive](#deployment-pipeline-deep-dive)
6. [Neural Network Architecture](#neural-network-architecture)
7. [Data Format & Structure](#data-format--structure)
8. [Modernization Plan](#modernization-plan)
9. [Implementation Roadmap](#implementation-roadmap)

---

## Executive Summary

**Agile Autonomy** is a deep learning system that enables quadrotor drones to navigate through cluttered environments (forests, obstacles) at high speeds (1-10 m/s) using vision-based control. The system uses a neural network (PlaNet) trained via imitation learning to predict collision-free trajectories from depth images and IMU data.

### Key Capabilities
- **Real-time trajectory prediction** at 15Hz
- **Multi-modal outputs** (3 trajectory candidates)
- **Collision-aware planning** using 3D point clouds
- **Speed range:** 1-10 m/s
- **Prediction horizon:** 1 second (10 steps @ 0.1s each)

### Current State
- **Built on:** ROS Noetic, TensorFlow 2.4, Ubuntu 20.04
- **Heavy ROS dependencies** for simulation, data collection, control
- **C++ components** for simulation and low-level control
- **Mixed pipeline** combining Python ML with C++ robotics stack

### Goal
Rebuild as **pure Python system** with modern dependencies for both training and deployment, removing all ROS dependencies while maintaining or improving functionality.

---

## Current Architecture Overview

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIMULATION ENVIRONMENT                        │
│  (Flightmare - C++/Unity, Vulkan Rendering)                     │
│  - Physics simulation                                            │
│  - Vision rendering (RGB/Depth)                                  │
│  - Environment generation (forests, obstacles)                   │
└────────────────┬────────────────────────────────────────────────┘
                 │ ZMQ Socket
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              ROS MIDDLEWARE LAYER                                │
│  - Message passing (topics/services)                             │
│  - Data recording (rosbag alternative)                           │
│  - Node coordination                                             │
└────┬──────────────────────────────────────────────────┬─────────┘
     │                                                    │
     ▼                                                    ▼
┌─────────────────────────┐              ┌──────────────────────────┐
│   DATA GENERATION       │              │   NEURAL NETWORK         │
│   (C++ Nodes)           │              │   (Python)               │
│                         │              │                          │
│ - agile_autonomy node   │              │ - PlaNet model           │
│ - MPPI expert           │              │ - Training pipeline      │
│ - Point cloud renderer  │              │ - Inference engine       │
│ - State estimator       │              │ - Data loader            │
└─────────────────────────┘              └──────────────────────────┘
```

### Directory Structure

```
agile_autonomy/
├── planner_learning/           # Python ML pipeline (KEEP & MODERNIZE)
│   ├── src/PlannerLearning/
│   │   ├── models/             # Core ML components
│   │   │   ├── nets.py         # PlaNet network architecture
│   │   │   ├── plan_learner.py # Training/inference engine
│   │   │   ├── data_loader.py  # Dataset management
│   │   │   ├── utils.py        # Loss functions
│   │   │   ├── geometry.py     # 3D math utilities
│   │   │   └── pose.py         # Pose transformations
│   │   ├── PlannerBase.py      # ROS node base (REMOVE)
│   │   └── PlannerLearning.py  # ROS integration (REMOVE)
│   ├── config/                 # Configuration files
│   │   ├── settings.py         # Settings parser
│   │   ├── train_settings.yaml
│   │   ├── test_settings.yaml
│   │   └── dagger_settings.yaml
│   ├── train.py                # Training script
│   ├── test_trajectories.py    # Testing script
│   ├── dagger_training.py      # DAGGER loop (needs ROS removal)
│   └── ckpt/                   # Pre-trained weights
│
├── data_generation/            # ROS-based C++ code (REPLACE)
│   ├── agile_autonomy/         # Main simulation node (C++)
│   ├── traj_sampler/           # MPPI expert (C++)
│   ├── rpg_flightmare/         # Simulator interface
│   ├── agile_autonomy_msgs/    # ROS messages
│   └── viz_utils/              # Python visualization (KEEP)
│
└── logging_utils/              # Metrics (KEEP)
```

---

## Current Dependencies Analysis

### Python Dependencies (planner_learning)

#### Core ML Stack
- **tensorflow-gpu==2.4** (2020 release, outdated)
  - Uses Keras 2.x API
  - TF 1.x compatibility layer still present
  - MobileNet from `tf.python.keras.applications`

#### Scientific Computing
- **numpy** (version unspecified)
- **scipy** (for spatial transformations, KD-tree)
- **pandas** (for CSV data loading)
- **opencv-python** (image processing)
- **open3d** (point cloud processing)
- **matplotlib** (visualization)

#### ROS Integration (TO REMOVE)
- **rospy** - ROS Python client
- **rospkg==1.2.3** - ROS package utilities
- **cv_bridge** - ROS-OpenCV bridge
- **geometry_msgs, nav_msgs, sensor_msgs, std_msgs** - ROS messages
- **quadrotor_msgs** - Custom quadrotor messages
- **agile_autonomy_msgs** - Project-specific messages

#### Other
- **pyquaternion** - Quaternion math
- **tqdm** - Progress bars
- **PyYAML** - Configuration parsing

### ROS/C++ Dependencies (data_generation)

#### ROS Ecosystem (ALL TO REPLACE)
- **ROS Noetic** (full install)
- **catkin** build system
- **roscpp, rospy** - ROS core
- **tf2** - Coordinate transformations
- **cv_bridge** - Image conversion
- **message_generation** - Message compilation

#### External ROS Packages (FROM dependencies.yaml)
1. **rpg_flightmare** - Vision simulator (ZMQ, Vulkan, Unity)
2. **rpg_quadrotor_control** - Low-level controller
3. **rpg_mpc** - Model Predictive Control
4. **rotors_simulator** - Quadrotor physics
5. **rpg_mpl_ros** - Motion planning library
6. **mav_comm** - MAV messaging
7. **minkindr/minkindr_ros** - SE(3) transformations
8. **eigen_catkin, glog_catkin, gflags_catkin** - Build wrappers

#### System Libraries
- **Eigen3** - Linear algebra (C++)
- **libzmqpp-dev** - ZeroMQ C++ bindings
- **libglfw3-dev, libglm-dev** - OpenGL
- **libvulkan1, vulkan-utils** - Vulkan graphics
- **libqglviewer-dev-qt5** - 3D visualization

#### Build Tools
- **gcc/g++ 7.5.0** (specific version requirement)
- **CMake** (via catkin)
- **catkin_tools**

---

## Training Pipeline Deep Dive

### Current Training Flow

```
┌──────────────────────────────────────────────────────────────────┐
│ STEP 1: DATA COLLECTION (Simulation)                             │
└──────────────────────────────────────────────────────────────────┘
                              ▼
    1. Launch Flightmare simulator (Unity standalone)
    2. Start ROS nodes:
       - agile_autonomy (main control node)
       - flightmare_bridge (sim interface)
       - sgm_gpu (stereo depth matching)
    3. Run expert (MPPI trajectory optimizer)
    4. Record rollouts:
       - RGB images (PNG) @ 640x480 → 15Hz
       - Depth images (TIF) @ 640x480 → 15Hz
       - Odometry (CSV): position, rotation matrix, velocity
       - Point clouds (PLY): 3D environment
       - Reference trajectories (CSV): expert plans

┌──────────────────────────────────────────────────────────────────┐
│ STEP 2: DATA LABELING (Post-processing)                          │
└──────────────────────────────────────────────────────────────────┘
                              ▼
    1. For each rollout frame:
       - Run MPPI optimizer (C++ traj_sampler node)
       - Generate optimal trajectory from current state
       - Evaluate trajectory cost (collision + smoothness)
       - Save labeled trajectory to CSV
    2. Parallel processing: 8 threads (generate_label_8.launch)

┌──────────────────────────────────────────────────────────────────┐
│ STEP 3: DATASET PREPARATION (Python)                             │
└──────────────────────────────────────────────────────────────────┘
                              ▼
    1. PlanDataset.discover_rollouts():
       - Scan train/val directories for rollout_* folders
       - Index all available frames
    2. Load and preprocess:
       - Resize images to 224x224
       - Convert quaternions to rotation matrices
       - Transform velocities to body frame (if configured)
       - Compute reference trajectory progress
       - Filter invalid samples
    3. Create TensorFlow dataset with batching (batch_size=8)

┌──────────────────────────────────────────────────────────────────┐
│ STEP 4: TRAINING (Python - TensorFlow)                           │
└──────────────────────────────────────────────────────────────────┘
                              ▼
    Training Loop (150 epochs):

    For each batch:
      1. Load inputs:
         - Depth images: (batch, seq_len=1, 224, 224, 3)
         - States: (batch, seq_len=1, 13)
           [position(3), attitude(9), velocity(3), body_rates(3)]
         - Reference trajectories: (batch, out_seq_len=10, 3)
         - Point clouds: (batch, N_points, 3)

      2. Forward pass:
         - PlaNet network inference
         - Output: (batch, modes=3, 31)
           [positions(30) + alpha(1) per mode]

      3. Loss computation:
         a) MixtureSpaceLoss:
            - Compare predictions to ground truth
            - Weight by mode probabilities (alpha)
         b) TrajectoryCostLoss:
            - Evaluate trajectories against point clouds
            - Penalize near-collision paths
         Total = space_loss + collision_cost_loss

      4. Backpropagation:
         - Adam optimizer (lr with cosine decay)
         - Gradient clipping

      5. Metrics tracking:
         - Training loss
         - Validation loss
         - Per-mode accuracy

      6. Checkpoint saving (every 5 epochs)

┌──────────────────────────────────────────────────────────────────┐
│ ALTERNATIVE: DAGGER TRAINING (Iterative Learning)                │
└──────────────────────────────────────────────────────────────────┘
                              ▼
    DAGGER Loop (max_rollouts times):

    1. Execute rollout:
       - Let network control drone (with expert fallback)
       - Reduce expert radius progressively

    2. Collect data:
       - Record state, images, depth

    3. Expert labeling:
       - Run MPPI on collected data
       - Generate optimal trajectories

    4. Aggregate dataset:
       - Add new labeled data to training set

    5. Retrain network (every N rollouts):
       - Use accumulated data
       - Continue from previous checkpoint

    6. Evaluate performance:
       - Track success rate, collisions
       - Adjust expert intervention threshold
```

### Training Configuration (train_settings.yaml)

```yaml
# Data
train_dir: "data/train"
val_dir: "data/val"

# Network inputs
use_rgb: False
use_depth: True
img_width: 224
img_height: 224
seq_len: 1                    # Temporal sequence length
state_dim: 3                  # x, y, z
out_seq_len: 10               # Predict 1 second (10 * 0.1s)

# Network architecture
modes: 3                      # Multi-modal predictions
inputs:
  position: False             # Don't use absolute position
  attitude: True              # Use rotation matrix (9 values)
  bodyrates: True             # Use angular velocities
  velocity_frame: 'bf'        # Body frame velocities

# Training hyperparameters
batch_size: 8
max_training_epochs: 150
freeze_backbone: False        # Train MobileNet too
save_every_n_epochs: 5
learning_rate: default        # Cosine decay
optimizer: Adam

# Reference tracking
ref_frame: 'bf'               # Body frame references
track_global_traj: False      # Don't follow global plan
future_time: 5.0              # Track point 5s ahead

# Loss configuration
top_trajectories: 3           # Use all modes
```

---

## Deployment Pipeline Deep Dive

### Current Inference Flow

```
┌──────────────────────────────────────────────────────────────────┐
│ RUNTIME SYSTEM ARCHITECTURE                                      │
└──────────────────────────────────────────────────────────────────┘

[Simulation/Real Drone]
         │
         ├─ Odometry (30Hz)
         ├─ RGB Camera (30Hz)
         ├─ Depth Sensor (30Hz)
         └─ Point Cloud (on demand)
         │
         ▼
[ROS Topics] ──────────────────────────┐
         │                             │
         ▼                             ▼
[PlannerBase.py]                 [agile_autonomy node]
         │                             │
         ├─ Queue management           ├─ Low-level control
         ├─ Image preprocessing        ├─ Trajectory execution
         └─ Network trigger (15Hz)     └─ Data recording
                 │
                 ▼
         [PlaNet Inference]
                 │
                 ├─ Input preparation
                 ├─ Forward pass
                 └─ Post-processing
                 │
                 ▼
         [Multi-Modal Trajectories]
                 │
                 ├─ Mode 1: x,y,z(10 steps) + alpha
                 ├─ Mode 2: x,y,z(10 steps) + alpha
                 └─ Mode 3: x,y,z(10 steps) + alpha
                 │
                 ▼
         [Trajectory Selection]
                 │
                 └─ Choose mode with lowest collision cost
                 │
                 ▼
         [Publish to ROS]
                 │
                 └─ MultiTrajectory message
                 │
                 ▼
         [Controller Execution]
```

### Inference Step-by-Step

```python
# 1. INITIALIZATION (once)
- Load checkpoint (ckpt-50)
- Build network graph
- Allocate GPU memory
- Initialize ROS subscribers/publishers
- Start sensor queues

# 2. SENSOR CALLBACK LOOP (30Hz)
def odometry_callback(msg):
    state_queue.append({
        'position': msg.pose.position,
        'orientation': msg.pose.orientation,  # quaternion
        'velocity': msg.twist.linear
    })

def depth_callback(msg):
    depth_img = bridge.imgmsg_to_cv2(msg)
    depth_queue.append(depth_img)

# 3. NETWORK TRIGGER (15Hz)
def network_timer_callback():
    # Get latest data
    state = state_queue[-1]
    depth = depth_queue[-1]

    # Prepare inputs
    depth_resized = cv2.resize(depth, (224, 224))
    depth_normalized = depth_resized / depth_resized.max()

    # Convert quaternion to rotation matrix
    rot_mat = quaternion_to_rotation_matrix(state['orientation'])

    # Transform velocity to body frame
    vel_body = rot_mat.T @ state['velocity']

    # Stack state vector [pos(3), rot(9), vel(3), bodyrates(3)]
    state_vec = np.concatenate([
        state['position'],      # 3
        rot_mat.flatten(),      # 9
        vel_body,               # 3
        state['bodyrates']      # 3 (from IMU)
    ])  # Total: 18 (but only 13 used based on config)

    # Inference
    predictions = network.inference(
        depth_img=depth_normalized,
        state=state_vec,
        reference_traj=reference  # straight line
    )

    # predictions shape: (modes=3, output_dim=31)
    # Each mode: [x,y,z * 10 steps, alpha]

    # Select best mode
    best_mode_idx = np.argmin(predictions[:, -1])  # Lowest alpha
    best_trajectory = predictions[best_mode_idx, :-1]

    # Reshape to (10, 3)
    trajectory_points = best_trajectory.reshape(10, 3)

    # Transform from body frame to world frame
    trajectory_world = []
    for point in trajectory_points:
        world_point = state['position'] + rot_mat @ point
        trajectory_world.append(world_point)

    # Publish
    publish_trajectory(trajectory_world)

# 4. TRAJECTORY EXECUTION (handled by C++ controller)
- Receives trajectory via ROS topic
- Computes motor commands via MPC
- Sends PWM to motors at 500Hz
```

### Test Configuration (test_settings.yaml)

```yaml
checkpoint:
  resume_file: "ckpt/ckpt-50"

# Inference settings
network_frequency: 15.0       # Hz
prediction_horizon: 10        # Steps
time_step: 0.1                # Seconds

# Environment
environment:
  avg_tree_spacing: 5.0       # Meters
  spawn_trees: True
  spawn_objects: False

# Speed
test_time_velocity: 7.0       # m/s

# Starting positions
start_poses:
  - [0, 0, 1.5, 0]            # x, y, z, yaw
  - [0, 0, 1.5, 1.57]
  # ... more start positions
```

---

## Neural Network Architecture

### PlaNet Network Detailed Breakdown

```
INPUT BRANCHES:
════════════════════════════════════════════════════════════════

1. IMAGE BRANCH (if use_depth or use_rgb):
   Input: (batch, seq_len=1, 224, 224, channels)
          channels = 3 (depth) or 3 (rgb) or 6 (both)

   ┌─────────────────────────────────────┐
   │ MobileNet Backbone (ImageNet)       │
   │ - Pretrained weights                │
   │ - Depthwise separable convolutions  │
   │ - Output: (batch, 7, 7, 1024)       │
   └─────────────────────────────────────┘
            ▼
   GlobalAveragePooling2D
            ▼ (batch, 1024)
   Expand dims → (batch, seq_len=1, 1024)
            ▼
   Conv1D(128, kernel=1)  # Reduce dimension
            ▼ (batch, 1, 128)
   Conv1D(128, k=2) + LeakyReLU
            ▼
   Conv1D(64, k=2) + LeakyReLU
            ▼
   Conv1D(64, k=2) + LeakyReLU
            ▼
   Conv1D(32, k=2) + LeakyReLU
            ▼ (batch, 1, 32)
   Conv1D(modes=3, k=3, padding=valid)
            ▼
   Output: img_features (batch, modes=3, 32)
            via reshape/transpose

2. STATE BRANCH (IMU data):
   Input: (batch, seq_len=1, state_features)

   state_features composition:
   - position (3): x, y, z           [if inputs.position=True]
   - attitude (9): rotation matrix   [if inputs.attitude=True]
   - velocity (3): vx, vy, vz        [always included]
     - Frame: body or world          [inputs.velocity_frame]
   - bodyrates (3): wx, wy, wz       [if inputs.bodyrates=True]

   Default config → 13 features:
   [attitude(9), velocity(3), bodyrates(3), altitude(1)]

   ┌─────────────────────────────────────┐
   │ State Processing Network            │
   └─────────────────────────────────────┘
            ▼
   Conv1D(64, k=2) + LeakyReLU(0.5)
            ▼
   Conv1D(32, k=2) + LeakyReLU(0.5)
            ▼
   Conv1D(32, k=2) + LeakyReLU(0.5)
            ▼
   Conv1D(32, k=2)
            ▼ (batch, 1, 32)
   Conv1D(modes=3, k=3, padding=valid)
            ▼
   Output: state_features (batch, modes=3, 32)

FUSION & PREDICTION:
════════════════════════════════════════════════════════════════

3. PLAN MODULE:
   Input: Concatenate [img_features, state_features]
          → (batch, modes=3, 64) if both enabled
          → (batch, modes=3, 32) if only one enabled

   ┌─────────────────────────────────────┐
   │ Multi-Modal Trajectory Predictor    │
   └─────────────────────────────────────┘
            ▼
   Conv1D(64, k=1) + LeakyReLU(0.5)
            ▼
   Conv1D(128, k=1) + LeakyReLU(0.5)
            ▼
   Conv1D(128, k=1) + LeakyReLU(0.5)
            ▼
   Conv1D(output_dim, k=1)
            ▼
   Output: (batch, modes=3, output_dim)

   output_dim = state_dim * out_seq_len + 1
              = 3 * 10 + 1 = 31

   Interpretation per mode:
   - positions[0:30]: x,y,z for 10 future steps
   - alpha[30]: mode probability/cost

OUTPUT STRUCTURE:
════════════════════════════════════════════════════════════════

Shape: (batch, 3, 31)

Mode 0: [x0, y0, z0, x1, y1, z1, ..., x9, y9, z9, alpha0]
Mode 1: [x0, y0, z0, x1, y1, z1, ..., x9, y9, z9, alpha1]
Mode 2: [x0, y0, z0, x1, y1, z1, ..., x9, y9, z9, alpha2]

Where:
- (xi, yi, zi): Position at time t + i*0.1s (in body frame)
- alphai: Mode selection weight (lower = better)
```

### Network Summary

```
Total Parameters: ~3.5M (with MobileNet)
- MobileNet: ~3.2M
- Custom layers: ~300K

Training:
- Optimizer: Adam
- Learning rate: Cosine decay (starts ~1e-3)
- Gradient clipping: norm = 1.0
- Batch size: 8
- Epochs: 150
- Checkpoint frequency: Every 5 epochs

Inference:
- Latency: ~30-50ms on GPU (GTX 1080)
- Frequency: 15Hz (66ms budget)
- Batch size: 1
```

---

## Data Format & Structure

### Rollout Directory Structure

```
data/
└── train/
    ├── rollout_21-09-21-1234/
    │   ├── left_images/
    │   │   ├── 000000.png
    │   │   ├── 000001.png
    │   │   └── ...
    │   ├── depth_images/
    │   │   ├── 000000.tif
    │   │   ├── 000001.tif
    │   │   └── ...
    │   ├── pointclouds/
    │   │   ├── 000000.ply
    │   │   ├── 000001.ply
    │   │   └── ...
    │   ├── odometry.csv
    │   ├── expert_trajectories.csv
    │   └── rollout_info.yaml
    ├── rollout_21-09-21-1235/
    └── ...
```

### Data File Formats

#### 1. odometry.csv
```csv
timestamp,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z,vel_x,vel_y,vel_z,omega_x,omega_y,omega_z
1632234567.123,0.0,0.0,1.5,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
1632234567.223,0.7,0.0,1.5,0.999,0.0,0.0,0.044,7.0,0.0,0.0,0.0,0.0,0.0
...
```

Columns:
- `timestamp`: Unix time (seconds)
- `pos_x, pos_y, pos_z`: Position in world frame (meters)
- `quat_w, quat_x, quat_y, quat_z`: Orientation (quaternion)
- `vel_x, vel_y, vel_z`: Velocity in world frame (m/s)
- `omega_x, omega_y, omega_z`: Body rates (rad/s)

#### 2. expert_trajectories.csv
```csv
timestamp,ref_x0,ref_y0,ref_z0,ref_x1,ref_y1,ref_z1,...,ref_x9,ref_y9,ref_z9
1632234567.123,1.0,0.0,1.5,2.0,0.0,1.5,...,10.0,0.0,1.5
...
```

Each row: 1 + 30 columns (timestamp + 10 waypoints * 3 coordinates)

#### 3. Images
- **RGB**: PNG format, 640×480, 8-bit per channel
- **Depth**: TIF format, 640×480, 16-bit or float32
  - Values in meters
  - 0 = invalid/infinity

#### 4. Point Clouds (PLY)
```
ply
format ascii 1.0
element vertex 12453
property float x
property float y
property float z
end_header
0.123 0.456 0.789
...
```

Used for:
- Collision cost computation during training
- Trajectory evaluation in TrajectoryCostLoss

#### 5. rollout_info.yaml
```yaml
start_time: 1632234567.0
end_time: 1632234589.5
num_frames: 338
success: true
collisions: 0
distance_traveled: 156.3
avg_speed: 7.2
environment:
  tree_spacing: 4.0
  spawn_trees: true
  spawn_objects: false
```

### Dataset Statistics (Example from Paper)

```
Training Set:
- Rollouts: 1,234
- Frames: ~416,000
- Total size: ~180 GB
- Environments: Forest (4-8m spacing)
- Speed range: 5-9 m/s
- Success rate: 87%

Validation Set:
- Rollouts: 123
- Frames: ~41,000
- Same environment distribution
```

---

## Modernization Plan

### Core Principle
**Remove all ROS dependencies while maintaining or improving functionality through modern Python libraries.**

---

## Phase 1: Modern Python Stack Selection

### 1.1 Core ML Framework

**Current:** TensorFlow 2.4 (2020)
**Modern Alternative:** PyTorch 2.1+ or TensorFlow 2.15+

**Recommendation: PyTorch 2.1**

Rationale:
- More Pythonic API
- Better debugging (eager execution by default)
- Stronger ecosystem for research
- JIT compilation (TorchScript) for deployment
- ONNX export for cross-platform deployment
- Active development and community

Migration considerations:
- Convert Keras layers to torch.nn modules
- Replace Conv1D with Conv1d
- Rewrite training loop (no model.fit())
- Update checkpoint format

**Alternative: Keep TensorFlow 2.15**
- Easier migration (mostly API compatible)
- Can use Keras 3.0 (multi-backend)
- Good for production deployment

### 1.2 Simulation Replacement

**Current:** Flightmare (Unity + Vulkan) + ROS

**Modern Alternatives:**

**Option A: PyBullet + Custom Rendering**
```python
import pybullet as p
import pybullet_data
```
- Pure Python physics
- Built-in quadrotor models
- GPU-accelerated rendering
- Camera sensors (RGB, depth)
- Easy installation: `pip install pybullet`

**Option B: Isaac Gym (NVIDIA)**
- GPU-accelerated physics
- Massive parallelization (1000+ envs)
- Realistic rendering
- Requires NVIDIA GPU
- Free for research

**Option C: MuJoCo + OpenGL**
- Industry-standard physics
- Now open-source
- Python bindings: `dm_control`, `mujoco`
- Excellent for robotics

**Recommendation: PyBullet for Phase 1**
- Easiest setup
- Cross-platform
- Sufficient for initial development
- Can upgrade to Isaac Gym later for scale

### 1.3 Sensor Simulation

| Sensor | Current | Modern Replacement |
|--------|---------|-------------------|
| RGB Camera | Flightmare/Unity | PyBullet `getCameraImage()` |
| Depth Camera | Stereo SGM (C++) | PyBullet depth buffer + NumPy |
| IMU | ROS sensor_msgs | Custom Python class |
| Point Cloud | PLY files | Open3D `PointCloud` |
| Odometry | ROS nav_msgs | Python dataclass |

### 1.4 Control & Planning

**Current:** C++ MPPI + ROS MPC

**Modern Replacement:**

```python
# Option A: Python MPPI
import mppi_torch  # GPU-accelerated

# Option B: Simple MPC
from scipy.optimize import minimize

# Option C: Differentiable physics
import torch
# Train end-to-end with differentiable simulator
```

**Recommendation:** Start with simple PID controller, add MPPI in Python later

### 1.5 Data Handling

| Function | Current | Modern |
|----------|---------|--------|
| Image I/O | cv_bridge (ROS) | OpenCV + Pillow |
| Depth | ROS messages | NumPy arrays (.npy) |
| Point clouds | ROS PointCloud2 | Open3D |
| Trajectories | CSV | HDF5 or Parquet |
| Datasets | Custom | PyTorch DataLoader |
| Recording | rosbag | HDF5 + JSON |

### 1.6 Coordinate Transformations

**Current:** tf2 (ROS), minkindr (C++)

**Modern:**
```python
from scipy.spatial.transform import Rotation
import numpy as np

# Or
from pytransform3d.rotations import *
```

### 1.7 Visualization

**Current:** RViz (ROS)

**Modern:**
```python
# 3D visualization
import open3d as o3d
import plotly.graph_objects as go

# Metrics
import wandb  # Weights & Biases
import tensorboard

# Videos
import cv2
import imageio
```

---

## Phase 2: Architecture Design

### 2.1 New Project Structure

```
agile_autonomy_v2/
├── README.md
├── requirements.txt
├── setup.py
├── pyproject.toml              # Modern Python packaging
│
├── config/
│   ├── train_config.yaml
│   ├── test_config.yaml
│   └── sim_config.yaml
│
├── agile_autonomy/             # Main package
│   ├── __init__.py
│   │
│   ├── models/                 # Neural networks
│   │   ├── __init__.py
│   │   ├── planet.py           # PlaNet architecture
│   │   ├── backbones.py        # MobileNet, etc.
│   │   └── losses.py           # Custom loss functions
│   │
│   ├── data/                   # Data handling
│   │   ├── __init__.py
│   │   ├── dataset.py          # PyTorch Dataset
│   │   ├── transforms.py       # Augmentations
│   │   └── loader.py           # DataLoader utils
│   │
│   ├── simulation/             # Simulator
│   │   ├── __init__.py
│   │   ├── quadrotor.py        # Drone model
│   │   ├── environment.py      # Forest generation
│   │   ├── sensors.py          # Camera, IMU, depth
│   │   └── controller.py       # Low-level control
│   │
│   ├── planning/               # Trajectory planning
│   │   ├── __init__.py
│   │   ├── mppi.py             # Expert planner
│   │   └── reference.py        # Reference trajectories
│   │
│   ├── utils/                  # Utilities
│   │   ├── __init__.py
│   │   ├── geometry.py         # 3D math
│   │   ├── transforms.py       # Coordinate frames
│   │   ├── visualization.py    # Plotting
│   │   └── metrics.py          # Evaluation
│   │
│   └── core/                   # Core abstractions
│       ├── __init__.py
│       ├── state.py            # State representation
│       ├── trajectory.py       # Trajectory class
│       └── config.py           # Configuration management
│
├── scripts/
│   ├── train.py                # Training script
│   ├── test.py                 # Evaluation
│   ├── collect_data.py         # Data collection
│   ├── visualize.py            # Visualization
│   └── export_model.py         # Model export (ONNX)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_analysis.ipynb
│   └── 03_results_visualization.ipynb
│
├── tests/
│   ├── test_models.py
│   ├── test_simulation.py
│   └── test_data.py
│
├── data/                       # Dataset (gitignored)
│   ├── raw/
│   ├── processed/
│   └── checkpoints/
│
└── docs/
    ├── architecture.md
    ├── training.md
    └── deployment.md
```

### 2.2 Modern Dependencies (requirements.txt)

```txt
# Core ML
torch>=2.1.0
torchvision>=0.16.0
# OR
# tensorflow>=2.15.0

# Simulation
pybullet>=3.2.5
gym>=0.26.0
# Optional: mujoco>=2.3.0

# Scientific Computing
numpy>=1.24.0
scipy>=1.11.0
pandas>=2.0.0
h5py>=3.9.0

# Computer Vision
opencv-python>=4.8.0
pillow>=10.0.0
open3d>=0.18.0

# 3D Math
pyquaternion>=1.1.2
pytransform3d>=3.3.0

# Visualization
matplotlib>=3.7.0
plotly>=5.17.0
open3d>=0.18.0
imageio>=2.31.0

# Experiment Tracking
tensorboard>=2.14.0
wandb>=0.15.0  # Optional but recommended

# Configuration
pyyaml>=6.0.0
omegaconf>=2.3.0  # Better config management
hydra-core>=1.3.0  # Optional

# Utilities
tqdm>=4.66.0
rich>=13.5.0  # Better CLI
typer>=0.9.0  # CLI framework

# Development
pytest>=7.4.0
black>=23.7.0
ruff>=0.0.287

# Optional: Hardware Deployment
# onnx>=1.14.0
# onnxruntime>=1.16.0
# jetson-gpio  # For NVIDIA Jetson
```

### 2.3 Configuration Management

**Modern approach with Hydra:**

```yaml
# config/train_config.yaml
defaults:
  - model: planet
  - dataset: forest
  - optimizer: adam

experiment:
  name: planet_forest_7ms
  seed: 42
  device: cuda

model:
  backbone: mobilenet_v2
  modes: 3
  prediction_horizon: 10
  use_depth: true
  use_rgb: false

training:
  epochs: 150
  batch_size: 8
  learning_rate: 1e-3
  scheduler: cosine
  gradient_clip: 1.0

dataset:
  train_dir: data/processed/train
  val_dir: data/processed/val
  image_size: [224, 224]
  augmentation: true

logging:
  use_wandb: true
  log_interval: 100
  checkpoint_interval: 5
```

---

## Phase 3: Implementation Roadmap

### Milestone 1: Core Infrastructure (Week 1-2)

**Tasks:**
1. Set up project structure
2. Install modern dependencies
3. Implement core data structures:
   - State representation
   - Trajectory class
   - Configuration system
4. Port geometry utilities (coordinate transforms)
5. Set up testing framework

**Deliverables:**
- Working project skeleton
- Unit tests for core utilities
- Documentation setup

### Milestone 2: Data Pipeline (Week 2-3)

**Tasks:**
1. Convert existing dataset format:
   - CSV → HDF5 for odometry
   - Keep images as PNG/TIF
   - Point clouds → Open3D format
2. Implement PyTorch Dataset:
   - Load rollouts
   - Preprocessing
   - Augmentation
3. Create DataLoader with batching
4. Visualization tools for data inspection

**Deliverables:**
- Dataset conversion scripts
- Working DataLoader
- Jupyter notebook for data exploration

### Milestone 3: Network Architecture (Week 3-4)

**Tasks:**
1. Port PlaNet to PyTorch:
   - Image branch (MobileNet)
   - State branch (Conv1d)
   - Plan module
2. Implement loss functions:
   - MixtureSpaceLoss
   - TrajectoryCostLoss
3. Add model tests
4. Implement checkpoint loading/saving

**Deliverables:**
- Complete PlaNet model in PyTorch
- Loss function tests
- Model architecture diagram

### Milestone 4: Training Pipeline (Week 4-5)

**Tasks:**
1. Implement training loop:
   - Forward/backward pass
   - Optimizer step
   - Learning rate scheduling
2. Add logging:
   - TensorBoard integration
   - W&B integration
   - Metric tracking
3. Checkpoint management
4. Validation loop

**Deliverables:**
- Complete training script
- Logging dashboard
- Trained model on existing data

### Milestone 5: Simulation Environment (Week 5-7)

**Tasks:**
1. Set up PyBullet:
   - Quadrotor model
   - Physics parameters
2. Implement sensors:
   - RGB camera
   - Depth camera
   - IMU simulation
3. Environment generation:
   - Procedural forest
   - Obstacle spawning
4. Simple controller (PID)

**Deliverables:**
- Working simulator
- Sensor outputs matching original
- Visualization of simulation

### Milestone 6: Expert Planner (Week 7-8)

**Tasks:**
1. Implement Python MPPI:
   - Trajectory sampling
   - Cost evaluation
   - Optimization loop
2. Integrate with simulator
3. Benchmark against original C++ version

**Deliverables:**
- Python MPPI implementation
- Performance comparison
- Expert trajectory generation

### Milestone 7: Data Collection (Week 8-9)

**Tasks:**
1. Implement data recording:
   - HDF5 writer
   - Image saving
   - Metadata logging
2. Rollout execution:
   - Expert control
   - Data capture
3. Dataset validation

**Deliverables:**
- Data collection scripts
- New dataset generated
- Quality validation

### Milestone 8: Inference Pipeline (Week 9-10)

**Tasks:**
1. Implement real-time inference:
   - Model loading
   - Input preprocessing
   - Output post-processing
2. Integration with simulator
3. Trajectory selection logic
4. Performance optimization

**Deliverables:**
- Inference script
- End-to-end flight in simulation
- Latency benchmarks

### Milestone 9: Evaluation & Validation (Week 10-11)

**Tasks:**
1. Implement test scenarios:
   - Different environments
   - Speed variations
   - Start positions
2. Metrics computation:
   - Success rate
   - Collision count
   - Trajectory smoothness
3. Comparison with original

**Deliverables:**
- Test suite
- Benchmark results
- Performance report

### Milestone 10: Documentation & Deployment (Week 11-12)

**Tasks:**
1. Complete documentation:
   - API docs
   - Training guide
   - Deployment guide
2. Model export:
   - ONNX format
   - TorchScript
3. Deployment examples:
   - CPU inference
   - Jetson Nano
4. Docker container

**Deliverables:**
- Full documentation
- Deployment package
- Docker image

---

## Phase 4: Advanced Features (Optional)

### 4.1 Performance Optimization
- [ ] Mixed precision training (FP16)
- [ ] Model quantization for edge devices
- [ ] Batch inference optimization
- [ ] GPU kernel fusion

### 4.2 Enhanced Simulation
- [ ] Upgrade to Isaac Gym for parallel training
- [ ] Realistic lighting variations
- [ ] Weather effects (fog, rain)
- [ ] Domain randomization

### 4.3 Improved Training
- [ ] Multi-task learning (speed prediction, obstacle detection)
- [ ] Self-supervised learning from unlabeled data
- [ ] Online DAGGER with improved exploration
- [ ] Uncertainty estimation

### 4.4 Real-World Deployment
- [ ] ROS2 bridge (optional, for hardware)
- [ ] Raspberry Pi / Jetson deployment
- [ ] Real-time video streaming
- [ ] Safety monitors (collision detection)

---

## Key Differences from Original

| Aspect | Original | Modern Python |
|--------|----------|---------------|
| **Build System** | Catkin (CMake) | pip/conda |
| **Language Mix** | C++ + Python | Pure Python |
| **ROS Dependency** | Essential | None |
| **ML Framework** | TF 2.4 | PyTorch 2.1 |
| **Simulator** | Flightmare (Unity) | PyBullet |
| **Data Format** | rosbag + CSV | HDF5 + PNG |
| **Expert** | C++ MPPI | Python MPPI |
| **Visualization** | RViz | Open3D + Plotly |
| **Deployment** | ROS nodes | Python scripts |
| **Configuration** | YAML (simple) | Hydra (advanced) |
| **Logging** | Custom | W&B + TensorBoard |

---

## Migration Strategy

### Option A: Gradual Migration (Safer)
1. Keep ROS environment running
2. Build Python-only components alongside
3. Test parity at each stage
4. Switch over when validated

### Option B: Clean Slate (Faster)
1. Extract essential logic from ROS code
2. Build pure Python from scratch
3. Validate with existing checkpoints
4. Test on converted dataset

**Recommendation: Option B** - Clean slate
- Original codebase is complex with ROS coupling
- Easier to maintain modern codebase
- Can reference original for validation
- Faster development without legacy constraints

---

## Success Criteria

### Functional Parity
- [ ] Same network architecture and performance
- [ ] Equivalent or better training speed
- [ ] Match or exceed original success rate (87%+)
- [ ] Real-time inference at 15Hz+

### Code Quality
- [ ] 100% Python, no ROS dependencies
- [ ] Type hints throughout
- [ ] >80% test coverage
- [ ] Clean architecture (modular, testable)

### Usability
- [ ] Simple installation (`pip install -e .`)
- [ ] Clear documentation
- [ ] Easy configuration
- [ ] Good debugging experience

### Performance
- [ ] Training: <24 hours for 150 epochs (single GPU)
- [ ] Inference: <50ms per prediction
- [ ] Memory: <4GB GPU for inference

---

## Estimated Timeline

**Total: 12 weeks (3 months) for full implementation**

- Weeks 1-2: Infrastructure
- Weeks 3-5: ML Pipeline (data + model + training)
- Weeks 6-8: Simulation + Expert
- Weeks 9-10: Integration + Inference
- Weeks 11-12: Testing + Documentation

**Fast-track: 6-8 weeks** (focused on core features only)

---

## Next Steps

1. **Approve this plan** and clarify priorities
2. **Set up development environment**
3. **Start with Milestone 1** (infrastructure)
4. **Parallel work**:
   - Convert dataset in background
   - Port network architecture
5. **Weekly checkpoints** to track progress

---

## Questions for Clarification

1. **Primary goal**: Research (flexibility) or deployment (performance)?
2. **Hardware target**: Workstation GPU, Jetson, or cloud?
3. **Timeline**: Aggressive (8 weeks) or thorough (12 weeks)?
4. **Simulator**: PyBullet (easy) or Isaac Gym (powerful)?
5. **ML framework**: PyTorch (preferred) or TensorFlow?
6. **Logging**: Just TensorBoard or also W&B?

---

## References

- Original Paper: [Loquercio et al., Science Robotics 2021](http://rpg.ifi.uzh.ch/docs/Loquercio21_Science.pdf)
- Project Page: http://rpg.ifi.uzh.ch/AgileAutonomy.html
- Dataset: https://zenodo.org/record/5517791
- Current Repo: https://github.com/uzh-rpg/agile_autonomy

---

**Document Version:** 1.0
**Last Updated:** 2025-11-09
**Status:** Ready for Implementation
