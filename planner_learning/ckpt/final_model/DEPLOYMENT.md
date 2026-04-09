# PlaNet Deployment Model

Trained by: Agile Autonomy training pipeline
Architecture: MobileNet backbone + Conv1D planning head
Framework: TensorFlow 2.21.0

## Model Files

| File | Description |
|------|-------------|
| saved_model/ | TF SavedModel (universal) |
| planet_model.tflite | Quantised TFLite for edge (Jetson / RPi) |

## Input Specification

| Key | Shape | Description |
|-----|-------|-------------|
| imu | [1, 1, 21] | State vector: pos(3) + rot_mat(9) + vel_bf(3) + omega(3) + goal_dir(3) |
| depth | [1, 1, 224, 224, 3] | Normalised depth image (depth_mm / 80, tiled to 3ch) |
| roll_id | [1] | Rollout ID (set to 1.0 at inference) |

## Output Specification

Shape: [1, 3, 31]
- Axis 1: 3 trajectory modes (multimodal distribution)
- Axis 2: [alpha, x1..x10, y1..y10, z1..z10] per mode
  - alpha: mode confidence (lower = better trajectory)
  - positions: body-frame waypoints at 0.1s intervals

## ROS Deployment

1. Source your catkin workspace
2. Load the TFLite or SavedModel from this directory
3. At 15 Hz, call the network with:
   - IMU state from odometry topic
   - Depth image from SGM stereo processor
4. Select mode with lowest alpha → send as trajectory to controller

## Physical Safety Notes

- This model was trained on synthetic data; always test thoroughly in simulation
- Start at low speeds (1-3 m/s) and gradually increase
- Keep a safety pilot ready to take over at all times
- The model predicts in body frame; account for latency in the control loop
