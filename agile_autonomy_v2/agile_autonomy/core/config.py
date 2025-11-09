"""Configuration management."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from pathlib import Path
import yaml
from omegaconf import OmegaConf, DictConfig


@dataclass
class Config:
    """
    Configuration container with support for nested configs and YAML loading.

    This uses OmegaConf for powerful configuration management with:
    - Hierarchical configs
    - Variable interpolation
    - Type validation
    - Easy merging
    """

    _config: DictConfig = field(default_factory=lambda: OmegaConf.create())

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Config object
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        config = cls()
        config._config = OmegaConf.create(data)
        return config

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        config = cls()
        config._config = OmegaConf.create(data)
        return config

    def merge(self, other: 'Config') -> 'Config':
        """Merge with another config (other takes priority)."""
        merged = OmegaConf.merge(self._config, other._config)
        config = Config()
        config._config = merged
        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with dot notation.

        Args:
            key: Key with dot notation (e.g., 'model.backbone')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        try:
            return OmegaConf.select(self._config, key, default=default)
        except Exception:
            return default

    def set(self, key: str, value: Any) -> None:
        """Set configuration value with dot notation."""
        OmegaConf.update(self._config, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dictionary."""
        return OmegaConf.to_container(self._config, resolve=True)

    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            OmegaConf.save(self._config, f)

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to config values."""
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        return self.get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow attribute-style setting of config values."""
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            self.set(name, value)

    def __repr__(self) -> str:
        return f"Config({OmegaConf.to_yaml(self._config)})"


# Default configurations
DEFAULT_TRAIN_CONFIG = {
    'experiment': {
        'name': 'planet_training',
        'seed': 42,
        'device': 'cuda'
    },
    'model': {
        'backbone': 'mobilenet_v2',
        'modes': 3,
        'prediction_horizon': 10,
        'use_depth': True,
        'use_rgb': False,
        'img_size': [224, 224]
    },
    'training': {
        'epochs': 150,
        'batch_size': 8,
        'learning_rate': 1e-3,
        'scheduler': 'cosine',
        'gradient_clip': 1.0,
        'save_every_n_epochs': 5
    },
    'dataset': {
        'train_dir': 'data/processed/train',
        'val_dir': 'data/processed/val',
        'augmentation': True
    },
    'logging': {
        'use_wandb': False,
        'log_interval': 100,
        'checkpoint_dir': 'data/checkpoints'
    }
}

DEFAULT_SIM_CONFIG = {
    'physics': {
        'timestep': 1.0 / 240.0,
        'gravity': -9.81,
        'max_steps': 10000
    },
    'quadrotor': {
        'mass': 0.775,
        'arm_length': 0.17,
        'max_thrust': 15.0,
        'max_torque': 0.5
    },
    'camera': {
        'width': 640,
        'height': 480,
        'fov': 90.0,
        'near': 0.1,
        'far': 100.0
    },
    'environment': {
        'type': 'forest',
        'tree_spacing': 5.0,
        'area_size': [100, 100],
        'num_trees': 100
    },
    'controller': {
        'position_p': 1.0,
        'position_d': 0.5,
        'attitude_p': 3.0,
        'attitude_d': 0.3
    }
}
