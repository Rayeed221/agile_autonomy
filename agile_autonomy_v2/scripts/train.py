#!/usr/bin/env python
"""
Training script for PlaNet model.

Usage:
    python scripts/train.py --config config/train_config.yaml
    python scripts/train.py --config config/train_config.yaml --resume data/checkpoints/checkpoint_epoch_050.pt
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from agile_autonomy.core import Config
from agile_autonomy.models import PlaNet
from agile_autonomy.data import create_dataloader
from agile_autonomy.utils import Trainer, set_seed


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train PlaNet model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu), overrides config"
    )
    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    config = Config.from_yaml(args.config)
    config_dict = config.to_dict()

    # Override device if specified
    if args.device is not None:
        config_dict['experiment']['device'] = args.device

    # Set seed for reproducibility
    seed = config_dict.get('experiment', {}).get('seed', 42)
    set_seed(seed)

    print("\n" + "="*60)
    print("AGILE AUTONOMY - TRAINING")
    print("="*60)
    print(f"Experiment: {config_dict.get('experiment', {}).get('name', 'unnamed')}")
    print(f"Seed: {seed}")
    print(f"Device: {config_dict.get('experiment', {}).get('device', 'cuda')}")
    print("="*60 + "\n")

    # Create model
    print("Creating model...")
    model_config = config_dict.get('model', {})
    model = PlaNet(model_config)

    # Create data loaders
    print("Creating data loaders...")
    dataset_config = config_dict.get('dataset', {})
    training_config = config_dict.get('training', {})

    # Merge model config into dataset config for consistency
    dataset_full_config = {**model_config, **dataset_config, **training_config}

    train_loader = create_dataloader(
        data_dir=dataset_config.get('train_dir', 'data/processed/train'),
        config=dataset_full_config,
        mode='train',
        batch_size=training_config.get('batch_size', 8),
        shuffle=True,
        num_workers=dataset_config.get('num_workers', 4)
    )

    val_loader = create_dataloader(
        data_dir=dataset_config.get('val_dir', 'data/processed/val'),
        config=dataset_full_config,
        mode='val',
        batch_size=training_config.get('batch_size', 8),
        shuffle=False,
        num_workers=dataset_config.get('num_workers', 4)
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config_dict
    )

    # Train
    print("\nStarting training...\n")
    try:
        trainer.train(resume_from=args.resume)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")

        # Save checkpoint
        trainer.checkpoint_manager.save(
            epoch=trainer.current_epoch,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            metrics=trainer.val_metrics.get_summary(),
            is_best=False
        )
        print("Checkpoint saved")

    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
