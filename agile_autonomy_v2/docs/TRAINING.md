# Training Guide

This guide covers how to train the PlaNet model from scratch or fine-tune from a checkpoint.

## Quick Start

```bash
# Train from scratch
python scripts/train.py --config config/train_config.yaml

# Resume from checkpoint
python scripts/train.py --config config/train_config.yaml --resume data/checkpoints/checkpoint_epoch_050.pt

# Test trained model
python scripts/test.py --config config/test_config.yaml --checkpoint data/checkpoints/best_model.pt
```

---

## Model Architecture

### PlaNet (Planning Network)

The PlaNet model predicts multi-modal future trajectories from:
- **Depth images** (224×224, 3 channels)
- **IMU/state data** (attitude, velocity, body rates)

**Architecture:**
1. **Image Branch**: MobileNetV2 backbone + Conv1D processing
2. **State Branch**: Conv1D layers for temporal processing
3. **Plan Module**: Fuses features and predicts 3 trajectory modes

**Output:** 3 trajectory modes, each with:
- 10 waypoints (x, y, z) @ 0.1s intervals = 1 second ahead
- Alpha value (mode selection weight, lower = better)

### Loss Functions

**MixtureSpaceLoss:**
- Weighted MSE between predicted and ground truth trajectories
- Uses predicted alphas for soft mode selection

**TrajectoryCostLoss:**
- Evaluates trajectories against 3D point clouds
- Penalizes near-collision paths using KD-tree search
- Teaches the network to predict collision costs

**CombinedLoss:**
```python
total_loss = mixture_weight * mixture_loss + collision_weight * collision_loss
```

---

## Data Requirements

### Directory Structure

```
data/
├── processed/
│   ├── train/
│   │   ├── rollout_21-09-21-0001/
│   │   │   ├── depth_images/
│   │   │   │   ├── 000000.tif
│   │   │   │   └── ...
│   │   │   ├── left_images/         # Optional RGB
│   │   │   │   ├── 000000.png
│   │   │   │   └── ...
│   │   │   ├── pointclouds/         # For collision loss
│   │   │   │   ├── 000000.ply
│   │   │   │   └── ...
│   │   │   ├── odometry.csv
│   │   │   └── expert_trajectories.csv
│   │   └── ...
│   └── val/
│       └── ... (same structure)
```

### File Formats

**odometry.csv:**
```csv
timestamp,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z,vel_x,vel_y,vel_z,omega_x,omega_y,omega_z
1632234567.123,0.0,0.0,1.5,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
...
```

**expert_trajectories.csv:**
```csv
timestamp,ref_x0,ref_y0,ref_z0,ref_x1,ref_y1,ref_z1,...,ref_x9,ref_y9,ref_z9
1632234567.123,1.0,0.0,1.5,2.0,0.0,1.5,...,10.0,0.0,1.5
...
```

### Alternative: HDF5 Format (Recommended)

For faster loading, convert to HDF5:
```python
# Coming soon: convert_to_hdf5.py script
```

HDF5 structure:
```
train.h5
├── rollout_0/
│   ├── depth_images: (N, 224, 224, 3)
│   ├── odometry: (N, 13)
│   ├── references: (N, 10, 3)
│   └── point_clouds: (N, variable, 3)
└── ...
```

---

## Configuration

### Training Config (config/train_config.yaml)

**Key Parameters:**

```yaml
model:
  use_depth: true          # Use depth images
  use_rgb: false           # Use RGB images
  modes: 3                 # Number of trajectory modes
  out_seq_len: 10         # Prediction horizon (steps)
  freeze_backbone: false   # Whether to freeze MobileNet

training:
  epochs: 150
  batch_size: 8
  learning_rate: 0.001
  scheduler: "cosine"      # Learning rate schedule
  warmup_epochs: 10
  gradient_clip: 1.0

  # Loss weights
  mixture_weight: 1.0
  collision_weight: 1.0

dataset:
  train_dir: "data/processed/train"
  val_dir: "data/processed/val"
  num_workers: 4           # Data loading workers

logging:
  use_tensorboard: true
  use_wandb: false         # Optional W&B integration
  checkpoint_dir: "data/checkpoints"
```

---

## Training from Scratch

### 1. Prepare Data

Collect rollout data using the simulator (see DATA_COLLECTION.md) or download pre-collected dataset.

### 2. Edit Configuration

```bash
nano config/train_config.yaml
```

Update:
- `dataset.train_dir` and `dataset.val_dir`
- `experiment.name` for this run
- Adjust hyperparameters if needed

### 3. Start Training

```bash
python scripts/train.py --config config/train_config.yaml
```

**Training Output:**
```
Model: PlaNet
  Trainable parameters: 3,542,987
  Device: cuda
  Optimizer: Adam (lr=0.001)
  Scheduler: cosine
  Epochs: 150
  Batch size: 8
  Train samples: 41,600
  Val samples: 4,100

Epoch 1/150 [Train]: 100%|████████| 5200/5200 [02:15<00:00, 38.42it/s]
Epoch 1/150 [Val]:   100%|████████| 512/512 [00:12<00:00, 42.11it/s]

Epoch 1/150
  Train: total_loss: 0.2345 | mixture_loss: 0.1876 | collision_loss: 0.0469
  Val:   total_loss: 0.1987 | mixture_loss: 0.1598 | collision_loss: 0.0389
  LR:    0.000100

Saved checkpoint: data/checkpoints/checkpoint_epoch_001.pt
...
```

### 4. Monitor Training

**TensorBoard:**
```bash
tensorboard --logdir runs
```

Open http://localhost:6006 to see:
- Loss curves
- Learning rate schedule
- Metrics over time

**Weights & Biases (Optional):**
```yaml
# In config/train_config.yaml
logging:
  use_wandb: true
  wandb_project: "agile-autonomy"
```

---

## Fine-Tuning

To fine-tune from a pre-trained checkpoint:

```bash
python scripts/train.py \
  --config config/train_config.yaml \
  --resume data/checkpoints/pretrained_checkpoint.pt
```

**Recommended settings for fine-tuning:**
```yaml
training:
  epochs: 50               # Fewer epochs
  learning_rate: 0.0001    # Lower learning rate
  freeze_backbone: true    # Freeze MobileNet (optional)
```

---

## Evaluation

### Test on Simulation

```bash
python scripts/test.py --config config/test_config.yaml
```

**Output:**
```
Running 10 test rollouts...

Testing: 100%|████████| 10/10 [05:23<00:00, 32.35s/it]
  Rollout 0: Success=True, Time=28.3s, Distance=156.2m
  Rollout 1: Success=True, Time=29.1s, Distance=168.5m
  ...

============================================================
RESULTS
============================================================
Success Rate: 87.0% (87/100)
Average Time: 27.4s
Average Distance: 159.3m
Total Collisions: 13
============================================================

Results saved to: results/planet_test_forest_results.json
```

### Metrics Computed

- **Success rate**: % of rollouts without collision
- **Collision count**: Total collisions across all rollouts
- **Average time**: Time per rollout
- **Average distance**: Distance traveled
- **Trajectory smoothness**: Jerk and acceleration metrics

---

## Troubleshooting

### Out of Memory (OOM)

**Solutions:**
1. Reduce batch size:
   ```yaml
   training:
     batch_size: 4  # Default: 8
   ```

2. Use gradient accumulation:
   ```python
   # Coming soon in trainer
   ```

3. Freeze backbone:
   ```yaml
   model:
     freeze_backbone: true
   ```

### Training Not Converging

**Check:**
1. Data quality - visualize samples
2. Learning rate - try lower (1e-4)
3. Loss weights - balance mixture and collision
4. Gradient clipping - ensure it's enabled

### Slow Training

**Optimize:**
1. Increase data workers:
   ```yaml
   dataset:
     num_workers: 8  # Default: 4
   ```

2. Use HDF5 format instead of directories

3. Pin memory:
   ```yaml
   dataset:
     pin_memory: true
   ```

4. Mixed precision training (coming soon)

---

## Advanced Topics

### Custom Loss Functions

Implement custom loss in `agile_autonomy/models/losses.py`:

```python
class MyCustomLoss(nn.Module):
    def forward(self, predictions, targets):
        # Your loss logic
        return loss
```

### Learning Rate Scheduling

Available schedulers:
- `cosine`: Cosine annealing with warmup (default)
- `step`: Step decay
- `exponential`: Exponential decay

### Early Stopping

Enable in config:
```yaml
training:
  use_early_stopping: true
  early_stopping_patience: 20  # Epochs
```

### Multi-GPU Training

Coming soon: DDP support

---

## Checkpoint Management

### Automatic Checkpoints

Saved every N epochs:
```yaml
training:
  save_every_n_epochs: 5
```

### Checkpoint Files

```
data/checkpoints/
├── checkpoint_epoch_005.pt
├── checkpoint_epoch_010.pt
├── ...
├── best_model.pt            # Best validation loss
└── training_config.json     # Config used for training
```

### Loading Checkpoints

**In Python:**
```python
from agile_autonomy.models import PlaNet
import torch

model = PlaNet(config)
checkpoint = torch.load("data/checkpoints/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
```

**Resume Training:**
```bash
python scripts/train.py --resume data/checkpoints/checkpoint_epoch_050.pt
```

---

## Performance Benchmarks

### Expected Training Time

**Hardware:** NVIDIA RTX 3080 (10GB)

| Dataset Size | Epochs | Time     |
|--------------|--------|----------|
| 10K samples  | 150    | ~4 hours |
| 50K samples  | 150    | ~18 hours|
| 200K samples | 150    | ~3 days  |

### Inference Speed

| Hardware      | Latency | Frequency |
|---------------|---------|-----------|
| RTX 3080      | 15ms    | 66 Hz     |
| GTX 1080      | 30ms    | 33 Hz     |
| CPU (i7-9700K)| 120ms   | 8 Hz      |

---

## Next Steps

1. **Collect more data** - See DATA_COLLECTION.md
2. **Experiment with hyperparameters** - Grid search
3. **Deploy to hardware** - See DEPLOYMENT.md
4. **Implement DAGGER** - Iterative improvement

---

## References

- Original Paper: [Loquercio et al., Science Robotics 2021](http://rpg.ifi.uzh.ch/docs/Loquercio21_Science.pdf)
- Project Page: http://rpg.ifi.uzh.ch/AgileAutonomy.html
- PyTorch Docs: https://pytorch.org/docs/stable/index.html
