"""Trainer class for training PlaNet model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
import time

from ..models import PlaNet, CombinedLoss
from .training import (
    MetricsTracker,
    CheckpointManager,
    EarlyStopping,
    CosineAnnealingWarmup,
    save_training_config,
    count_parameters,
    get_device,
    set_seed
)


class Trainer:
    """
    Trainer for PlaNet model.

    Handles training loop, validation, logging, and checkpointing.
    """

    def __init__(
        self,
        model: PlaNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        """
        Initialize trainer.

        Args:
            model: PlaNet model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Device
        if device is None:
            device = get_device(use_cuda=config.get('device', 'cuda') == 'cuda')
        self.device = device
        self.model.to(self.device)

        # Training config
        training_config = config.get('training', {})
        self.epochs = training_config.get('epochs', 150)
        self.grad_clip = training_config.get('gradient_clip', 1.0)
        self.log_interval = config.get('logging', {}).get('log_interval', 100)

        # Loss function
        self.criterion = CombinedLoss(
            mixture_weight=training_config.get('mixture_weight', 1.0),
            collision_weight=training_config.get('collision_weight', 1.0),
            quadrotor_radius=training_config.get('quadrotor_radius', 0.3),
            environment_threshold=training_config.get('environment_threshold', 0.8)
        )

        # Optimizer
        lr = training_config.get('learning_rate', 1e-3)
        weight_decay = training_config.get('weight_decay', 0.0)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        scheduler_type = training_config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            warmup_epochs = training_config.get('warmup_epochs', 10)
            self.scheduler = CosineAnnealingWarmup(
                self.optimizer,
                warmup_epochs=warmup_epochs,
                max_epochs=self.epochs,
                eta_min=training_config.get('eta_min', 1e-6)
            )
        else:
            self.scheduler = None

        # Metrics tracking
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()

        # Checkpoint manager
        checkpoint_dir = config.get('logging', {}).get('checkpoint_dir', 'data/checkpoints')
        save_every_n_epochs = training_config.get('save_every_n_epochs', 5)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=5,
            save_best=True
        )
        self.save_every_n_epochs = save_every_n_epochs

        # Early stopping
        use_early_stopping = training_config.get('use_early_stopping', False)
        if use_early_stopping:
            patience = training_config.get('early_stopping_patience', 20)
            self.early_stopping = EarlyStopping(patience=patience, mode='min')
        else:
            self.early_stopping = None

        # TensorBoard
        use_tensorboard = config.get('logging', {}).get('use_tensorboard', True)
        if use_tensorboard:
            log_dir = config.get('logging', {}).get('tensorboard_dir', 'runs')
            experiment_name = config.get('experiment', {}).get('name', 'planet_training')
            self.writer = SummaryWriter(log_dir=f"{log_dir}/{experiment_name}")
        else:
            self.writer = None

        # Weights & Biases
        use_wandb = config.get('logging', {}).get('use_wandb', False)
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.get('logging', {}).get('wandb_project', 'agile-autonomy'),
                    name=experiment_name,
                    config=config
                )
                self.use_wandb = True
            except ImportError:
                print("Warning: wandb not installed, skipping W&B logging")
                self.use_wandb = False
        else:
            self.use_wandb = False

        # Print model info
        num_params = count_parameters(model)
        print(f"\nModel: PlaNet")
        print(f"  Trainable parameters: {num_params:,}")
        print(f"  Device: {self.device}")
        print(f"  Optimizer: Adam (lr={lr})")
        print(f"  Scheduler: {scheduler_type}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {train_loader.batch_size}")
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Val samples: {len(val_loader.dataset)}\n")

        # Current epoch
        self.current_epoch = 0

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}/{self.epochs} [Train]")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            depth = batch['depth'].to(self.device)  # (batch, seq_len, H, W, C)
            state = batch['state'].to(self.device)  # (batch, seq_len, state_dim)
            reference = batch['reference'].to(self.device)  # (batch, out_seq_len, 3)

            # Optional RGB
            rgb = batch.get('rgb')
            if rgb is not None:
                rgb = rgb.to(self.device)

            # Prepare image input (combine RGB and depth if both present)
            if self.model.use_rgb and self.model.use_depth:
                # Concatenate RGB and depth
                image = torch.cat([rgb, depth], dim=-1)  # (batch, seq_len, H, W, 6)
            elif self.model.use_rgb:
                image = rgb
            elif self.model.use_depth:
                image = depth
            else:
                image = None

            # Permute image to (batch, seq_len, channels, H, W)
            if image is not None:
                image = image.permute(0, 1, 4, 2, 3)

            # Forward pass
            predictions = self.model(state, image)  # (batch, modes, output_dim)

            # Flatten reference for loss
            batch_size = reference.shape[0]
            reference_flat = reference.reshape(batch_size, -1)  # (batch, out_seq_len * 3)

            # Compute loss
            point_clouds = batch.get('point_cloud')
            current_positions = batch.get('current_position')

            # Convert point clouds to numpy for KD-tree
            if point_clouds is not None:
                point_clouds_np = [pc.cpu().numpy() if pc is not None else None for pc in point_clouds]
            else:
                point_clouds_np = None

            losses = self.criterion(
                predictions,
                reference_flat,
                point_clouds=point_clouds_np,
                current_positions=current_positions
            )

            total_loss = losses['total']

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            # Update metrics
            self.train_metrics.update('total_loss', total_loss.item(), batch_size)
            self.train_metrics.update('mixture_loss', losses['mixture'].item(), batch_size)
            self.train_metrics.update('collision_loss', losses['collision'].item(), batch_size)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'mix': f"{losses['mixture'].item():.4f}",
                'col': f"{losses['collision'].item():.4f}"
            })

            # Log to TensorBoard
            if self.writer is not None and batch_idx % self.log_interval == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/total_loss', total_loss.item(), global_step)
                self.writer.add_scalar('train/mixture_loss', losses['mixture'].item(), global_step)
                self.writer.add_scalar('train/collision_loss', losses['collision'].item(), global_step)
                self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], global_step)

        return self.train_metrics.get_summary()

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        self.val_metrics.reset()

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch}/{self.epochs} [Val]  ")

        for batch in pbar:
            # Move batch to device
            depth = batch['depth'].to(self.device)
            state = batch['state'].to(self.device)
            reference = batch['reference'].to(self.device)

            rgb = batch.get('rgb')
            if rgb is not None:
                rgb = rgb.to(self.device)

            # Prepare image
            if self.model.use_rgb and self.model.use_depth:
                image = torch.cat([rgb, depth], dim=-1)
            elif self.model.use_rgb:
                image = rgb
            elif self.model.use_depth:
                image = depth
            else:
                image = None

            if image is not None:
                image = image.permute(0, 1, 4, 2, 3)

            # Forward pass
            predictions = self.model(state, image)

            # Flatten reference
            batch_size = reference.shape[0]
            reference_flat = reference.reshape(batch_size, -1)

            # Compute loss
            point_clouds = batch.get('point_cloud')
            current_positions = batch.get('current_position')

            if point_clouds is not None:
                point_clouds_np = [pc.cpu().numpy() if pc is not None else None for pc in point_clouds]
            else:
                point_clouds_np = None

            losses = self.criterion(
                predictions,
                reference_flat,
                point_clouds=point_clouds_np,
                current_positions=current_positions
            )

            # Update metrics
            self.val_metrics.update('total_loss', losses['total'].item(), batch_size)
            self.val_metrics.update('mixture_loss', losses['mixture'].item(), batch_size)
            self.val_metrics.update('collision_loss', losses['collision'].item(), batch_size)

            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}"
            })

        return self.val_metrics.get_summary()

    def train(self, resume_from: Optional[str] = None):
        """
        Run full training loop.

        Args:
            resume_from: Path to checkpoint to resume from
        """
        # Resume if requested
        if resume_from is not None:
            checkpoint = self.checkpoint_manager.load(
                resume_from,
                self.model,
                self.optimizer,
                self.scheduler
            )
            self.current_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {self.current_epoch}")

        # Save config
        save_training_config(self.config, self.checkpoint_manager.checkpoint_dir)

        best_val_loss = float('inf')

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Save metrics history
            self.train_metrics.save_epoch()
            self.val_metrics.save_epoch()

            # Print summary
            print(f"\nEpoch {epoch}/{self.epochs}")
            print(f"  Train: {self.train_metrics}")
            print(f"  Val:   {self.val_metrics}")
            print(f"  LR:    {self.optimizer.param_groups[0]['lr']:.6f}")

            # Log to TensorBoard
            if self.writer is not None:
                for key, value in train_metrics.items():
                    self.writer.add_scalar(f'epoch/train_{key}', value, epoch)
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'epoch/val_{key}', value, epoch)

            # Log to W&B
            if self.use_wandb:
                import wandb
                log_dict = {f'train/{k}': v for k, v in train_metrics.items()}
                log_dict.update({f'val/{k}': v for k, v in val_metrics.items()})
                log_dict['epoch'] = epoch
                log_dict['learning_rate'] = self.optimizer.param_groups[0]['lr']
                wandb.log(log_dict)

            # Save checkpoint
            val_loss = val_metrics['total_loss']
            is_best = val_loss < best_val_loss

            if is_best:
                best_val_loss = val_loss

            if (epoch + 1) % self.save_every_n_epochs == 0 or is_best:
                self.checkpoint_manager.save(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metrics=val_metrics,
                    is_best=is_best
                )

            # Early stopping check
            if self.early_stopping is not None:
                if self.early_stopping.step(val_loss):
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        # Training complete
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best checkpoint: {self.checkpoint_manager.get_best_checkpoint()}")
        print("="*60)

        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()

        # Finish W&B
        if self.use_wandb:
            import wandb
            wandb.finish()
