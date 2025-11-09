"""Training utilities (metrics, schedulers, checkpointing)."""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class MetricsTracker:
    """Track multiple metrics during training."""

    def __init__(self):
        self.metrics = {}
        self.history = {}

    def add(self, name: str):
        """Add a new metric to track."""
        self.metrics[name] = AverageMeter(name)
        self.history[name] = []

    def update(self, name: str, value: float, n: int = 1):
        """Update a metric."""
        if name not in self.metrics:
            self.add(name)
        self.metrics[name].update(value, n)

    def reset(self):
        """Reset all meters."""
        for meter in self.metrics.values():
            meter.reset()

    def get_average(self, name: str) -> float:
        """Get average value of a metric."""
        return self.metrics[name].avg if name in self.metrics else 0.0

    def save_epoch(self):
        """Save current averages to history."""
        for name, meter in self.metrics.items():
            self.history[name].append(meter.avg)

    def get_summary(self) -> Dict[str, float]:
        """Get summary of all metrics."""
        return {name: meter.avg for name, meter in self.metrics.items()}

    def __str__(self):
        return " | ".join([str(meter) for meter in self.metrics.values()])


class CosineAnnealingWarmup(_LRScheduler):
    """
    Cosine annealing learning rate scheduler with warm-up.

    Linearly increases LR from 0 to base_lr during warmup,
    then follows cosine annealing.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        eta_min: float = 0,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [
                self.eta_min + (base_lr - self.eta_min) * (1 + np.cos(np.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]


class CheckpointManager:
    """Manage model checkpoints."""

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_best: bool = True
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best: Whether to save best model separately
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.save_best = save_best

        self.checkpoints = []
        self.best_metric = float('inf')
        self.best_checkpoint = None

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            metrics: Dictionary of metrics
            is_best: Whether this is the best model

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # Save regular checkpoint
        filename = f"checkpoint_epoch_{epoch:03d}.pt"
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)

        # Track checkpoints
        self.checkpoints.append(filepath)

        # Remove old checkpoints if exceeding max
        if len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()

        # Save best model separately
        if is_best and self.save_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.best_checkpoint = best_path

        print(f"Saved checkpoint: {filepath}")
        if is_best:
            print(f"  → New best model!")

        return filepath

    def load(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optional optimizer to load state
            scheduler: Optional scheduler to load state

        Returns:
            Dictionary with checkpoint information
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"  Metrics: {checkpoint.get('metrics', {})}")

        return checkpoint

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to the latest checkpoint."""
        if not self.checkpoints:
            # Check directory for any checkpoints
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if checkpoints:
                return checkpoints[-1]
            return None
        return self.checkpoints[-1]

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to the best checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            return best_path
        return None


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

        if mode == 'min':
            self.is_better = lambda score, best: score < best - min_delta
        else:
            self.is_better = lambda score, best: score > best + min_delta

    def step(self, metric: float) -> bool:
        """
        Check if should stop.

        Args:
            metric: Current validation metric

        Returns:
            True if should stop training
        """
        if self.best_score is None:
            self.best_score = metric
            return False

        if self.is_better(metric, self.best_score):
            # Improvement
            self.best_score = metric
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"Early stopping triggered after {self.counter} epochs without improvement")
                return True

        return False


def save_training_config(config: Dict[str, Any], save_dir: str):
    """Save training configuration to JSON."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config_path = save_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Saved training config to {config_path}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(use_cuda: bool = True) -> torch.device:
    """Get device (CUDA if available and requested)."""
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print(f"Set random seed to {seed}")
