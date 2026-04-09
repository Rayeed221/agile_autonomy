#!/usr/bin/env python3
"""
Export final trained PlaNet model to multiple formats for physical deployment:

  1. TensorFlow SavedModel  → universal format, deploy with TF Serving / tflite
  2. TFLite FlatBuffer      → optimised for edge hardware (Jetson, RPi, etc.)
  3. ONNX                   → optional, broadest hardware support

Usage (after training):
    python export_model.py --ckpt_dir /tmp/agile_autonomy_final_model/TIMESTAMP/train
                           --out_dir  /tmp/final_model_export
                           [--tflite]
                           [--onnx]

The checkpoint directory is printed by train.py as
  "Saved checkpoint for epoch N: <path>/ckpt-K"
Run this script with the *directory* containing the ckpt-K files.
"""

import os, sys, argparse, glob
import numpy as np
import tensorflow as tf
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), 'src/PlannerLearning/models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))


# ─── Argument parsing ─────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description='Export trained PlaNet model')
    p.add_argument('--ckpt_dir',  required=True,
                   help='Directory containing TF checkpoint files (ckpt-K.*)')
    p.add_argument('--out_dir',   default='/tmp/final_model_export',
                   help='Output directory for exported files')
    p.add_argument('--tflite',    action='store_true', default=True,
                   help='Also export TFLite model (default: True)')
    p.add_argument('--onnx',      action='store_true', default=False,
                   help='Also export ONNX model (requires tf2onnx)')
    p.add_argument('--settings_file', default='config/train_settings.yaml',
                   help='Settings YAML used for training')
    return p.parse_args()


# ─── Minimal settings class (no log creation) ────────────────────────────────
class MinimalSettings:
    def __init__(self, yaml_file):
        with open(yaml_file) as f:
            s = yaml.safe_load(f)
        self.use_rgb             = s['use_rgb']
        self.use_depth           = s['use_depth']
        self.img_width           = s['img_width']
        self.img_height          = s['img_height']
        self.state_dim           = s['state_dim']
        self.out_seq_len         = s['out_seq_len']
        self.predict_state_number = s['predict_state_number']
        self.modes               = s['modes']
        self.seq_len             = s['seq_len']
        self.use_position        = s['inputs']['position']
        self.use_attitude        = s['inputs']['attitude']
        self.use_bodyrates       = s['inputs']['bodyrates']
        self.velocity_frame      = s['inputs']['velocity_frame']
        self.resume_training     = False
        self.resume_ckpt_file    = ''
        self.freeze_backbone     = s['train']['freeze_backbone']
        self.log_dir             = '/tmp/export_log_dummy'
        os.makedirs(self.log_dir, exist_ok=True)


# ─── Input specs for concrete function tracing ───────────────────────────────
def build_dummy_inputs(config, batch_size=1):
    """Build dummy input tensors matching the data_loader output."""
    imu_dim = 3 + 9 + 3    # pos + rot_matrix + vel = 15
    if config.use_attitude:
        pass               # already 15
    if config.use_bodyrates:
        imu_dim += 3       # add omega
    imu_dim += 3           # goal direction

    dummy = {
        'imu':     tf.zeros((batch_size, config.seq_len, imu_dim), dtype=tf.float32),
        'roll_id': tf.zeros((batch_size,),                         dtype=tf.float32),
    }
    if config.use_depth:
        dummy['depth'] = tf.zeros(
            (batch_size, config.seq_len, config.img_height, config.img_width, 3),
            dtype=tf.float32)
    if config.use_rgb:
        dummy['rgb'] = tf.zeros(
            (batch_size, config.seq_len, config.img_height, config.img_width, 3),
            dtype=tf.float32)
    return dummy


# ─── Main export logic ────────────────────────────────────────────────────────
def main():
    args   = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("="*60)
    print("PlaNet Model Export")
    print(f"  checkpoint  : {args.ckpt_dir}")
    print(f"  output      : {args.out_dir}")
    print("="*60)

    # ── Load settings ──────────────────────────────────────────────────────
    config = MinimalSettings(args.settings_file)

    # ── Rebuild network + restore checkpoint ──────────────────────────────
    from nets import create_network
    network = create_network(config)

    # Warm-up forward pass to build variables
    dummy = build_dummy_inputs(config)
    _ = network(dummy, training=False)
    print(f"  Network parameters : {network.count_params():,}")

    # Restore checkpoint
    ckpt = tf.train.Checkpoint(net=network)
    latest = tf.train.latest_checkpoint(args.ckpt_dir)
    if latest is None:
        print(f"ERROR: no checkpoint found in {args.ckpt_dir}")
        sys.exit(1)
    status = ckpt.restore(latest).expect_partial()
    print(f"  Restored checkpoint: {latest}")

    # ── TF SavedModel ──────────────────────────────────────────────────────
    saved_model_path = os.path.join(args.out_dir, 'saved_model')
    print(f"\nSaving TF SavedModel → {saved_model_path}")

    @tf.function(input_signature=[
        {k: tf.TensorSpec(shape=v.shape, dtype=v.dtype, name=k)
         for k, v in dummy.items()}
    ])
    def serve(inputs):
        return network(inputs, training=False)

    network.save(saved_model_path, save_format='tf',
                 signatures={'serving_default': serve})
    print("  SavedModel written.")

    # ── TFLite conversion ─────────────────────────────────────────────────
    if args.tflite:
        print("\nConverting to TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        tflite_path  = os.path.join(args.out_dir, 'planet_model.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        size_mb = os.path.getsize(tflite_path) / 1e6
        print(f"  TFLite model saved → {tflite_path}  ({size_mb:.1f} MB)")

    # ── Optional ONNX ────────────────────────────────────────────────────
    if args.onnx:
        try:
            import tf2onnx, onnx
            onnx_path = os.path.join(args.out_dir, 'planet_model.onnx')
            model_proto, _ = tf2onnx.convert.from_function(
                serve, input_signature=list(serve.input_signature))
            with open(onnx_path, 'wb') as f:
                f.write(model_proto.SerializeToString())
            print(f"  ONNX model saved → {onnx_path}")
        except ImportError:
            print("  Skipping ONNX (tf2onnx not installed). "
                  "Run: pip install tf2onnx")

    # ── Quick inference sanity check ──────────────────────────────────────
    print("\nRunning inference sanity check...")
    output = network(dummy, training=False).numpy()
    print(f"  Output shape : {output.shape}  "
          f"(expected: [1, {config.modes}, {config.state_dim * config.out_seq_len + 1}])")
    alphas = output[0, :, 0]
    print(f"  Mode alphas  : {alphas.round(4)}")

    # ── Write deployment README ───────────────────────────────────────────
    readme = f"""# PlaNet Deployment Model

Trained by: Agile Autonomy training pipeline
Architecture: MobileNet backbone + Conv1D planning head
Framework: TensorFlow {tf.__version__}

## Model Files

| File | Description |
|------|-------------|
| saved_model/ | TF SavedModel (universal) |
| planet_model.tflite | Quantised TFLite for edge (Jetson / RPi) |

## Input Specification

| Key | Shape | Description |
|-----|-------|-------------|
| imu | [1, 1, {dummy['imu'].shape[-1]}] | State vector: pos(3) + rot_mat(9) + vel_bf(3) + omega(3) + goal_dir(3) |
{'| depth | [1, 1, ' + str(config.img_height) + ', ' + str(config.img_width) + ', 3] | Normalised depth image (depth_mm / 80, tiled to 3ch) |' if config.use_depth else ''}
| roll_id | [1] | Rollout ID (set to 1.0 at inference) |

## Output Specification

Shape: [1, {config.modes}, {config.state_dim * config.out_seq_len + 1}]
- Axis 1: {config.modes} trajectory modes (multimodal distribution)
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
"""
    with open(os.path.join(args.out_dir, 'DEPLOYMENT.md'), 'w') as f:
        f.write(readme)

    print(f"\n{'='*60}")
    print("Export complete!")
    print(f"  Output directory: {args.out_dir}")
    print("  Files:")
    for fn in sorted(os.listdir(args.out_dir)):
        fpath = os.path.join(args.out_dir, fn)
        if os.path.isfile(fpath):
            print(f"    {fn} ({os.path.getsize(fpath)/1e6:.1f} MB)")
        else:
            print(f"    {fn}/")
    print("="*60)


if __name__ == '__main__':
    main()
