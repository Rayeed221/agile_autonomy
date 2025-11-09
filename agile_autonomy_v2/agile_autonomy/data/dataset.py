"""Dataset classes for loading rollout data."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
from scipy.spatial.transform import Rotation
import json


class RolloutDataset(Dataset):
    """
    Dataset for loading rollout data.

    Supports both HDF5 and directory-based formats.
    Each sample contains:
    - depth_image: (height, width) or (height, width, 3)
    - rgb_image: (height, width, 3) (optional)
    - state: state vector with position, attitude, velocity, etc.
    - reference_trajectory: ground truth trajectory
    - point_cloud: 3D point cloud (optional)
    - current_position: for point cloud queries
    """

    def __init__(
        self,
        data_dir: str,
        config: Dict[str, Any],
        mode: str = 'train',
        transform: Optional[Any] = None
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Root directory containing rollout data
            config: Configuration dictionary
            mode: 'train' or 'val'
            transform: Optional transformations
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.mode = mode
        self.transform = transform

        # Configuration
        self.use_rgb = config.get('use_rgb', False)
        self.use_depth = config.get('use_depth', True)
        self.img_height = config.get('img_height', 224)
        self.img_width = config.get('img_width', 224)
        self.out_seq_len = config.get('out_seq_len', 10)
        self.seq_len = config.get('seq_len', 1)

        # State configuration
        inputs_config = config.get('inputs', {})
        self.use_position = inputs_config.get('position', False)
        self.use_attitude = inputs_config.get('attitude', True)
        self.use_bodyrates = inputs_config.get('bodyrates', True)
        self.velocity_frame = inputs_config.get('velocity_frame', 'bf')  # 'bf' or 'world'

        # Load point clouds for collision loss
        self.load_point_clouds = config.get('use_collision_loss', True)

        # Discover rollouts
        self.rollouts = self._discover_rollouts()
        self.samples = self._index_samples()

        print(f"Found {len(self.rollouts)} rollouts with {len(self.samples)} samples")

    def _discover_rollouts(self) -> List[Path]:
        """Discover all rollout directories."""
        rollouts = []

        # Check if using HDF5 format
        hdf5_file = self.data_dir / f"{self.mode}.h5"
        if hdf5_file.exists():
            self.use_hdf5 = True
            self.hdf5_path = hdf5_file
            return []

        self.use_hdf5 = False

        # Look for rollout_* directories
        for rollout_dir in sorted(self.data_dir.glob("rollout_*")):
            if rollout_dir.is_dir():
                # Check if it has required files
                if (rollout_dir / "odometry.csv").exists():
                    rollouts.append(rollout_dir)

        return rollouts

    def _index_samples(self) -> List[Dict[str, Any]]:
        """Index all valid samples."""
        samples = []

        if self.use_hdf5:
            # Index HDF5 file
            with h5py.File(self.hdf5_path, 'r') as f:
                num_rollouts = len(f.keys())
                for rollout_idx in range(num_rollouts):
                    rollout_key = f"rollout_{rollout_idx}"
                    if rollout_key not in f:
                        continue

                    rollout_group = f[rollout_key]
                    num_frames = len(rollout_group['depth_images'])

                    for frame_idx in range(num_frames):
                        samples.append({
                            'rollout_idx': rollout_idx,
                            'frame_idx': frame_idx,
                            'source': 'hdf5'
                        })
        else:
            # Index directory-based format
            for rollout_dir in self.rollouts:
                # Count available frames
                depth_dir = rollout_dir / "depth_images"
                if not depth_dir.exists():
                    continue

                depth_files = sorted(depth_dir.glob("*.tif"))
                num_frames = len(depth_files)

                for frame_idx in range(num_frames):
                    samples.append({
                        'rollout_dir': rollout_dir,
                        'frame_idx': frame_idx,
                        'source': 'directory'
                    })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with keys:
            - 'depth': (seq_len, H, W, 3) or (seq_len, H, W, 1)
            - 'rgb': (seq_len, H, W, 3) if use_rgb
            - 'state': (seq_len, state_dim)
            - 'reference': (out_seq_len, 3)
            - 'point_cloud': (N, 3) or None
            - 'current_position': (3,)
        """
        sample_info = self.samples[idx]

        if sample_info['source'] == 'hdf5':
            return self._load_from_hdf5(sample_info)
        else:
            return self._load_from_directory(sample_info)

    def _load_from_hdf5(self, sample_info: Dict) -> Dict[str, torch.Tensor]:
        """Load sample from HDF5 file."""
        with h5py.File(self.hdf5_path, 'r') as f:
            rollout_key = f"rollout_{sample_info['rollout_idx']}"
            rollout_group = f[rollout_key]
            frame_idx = sample_info['frame_idx']

            # Load depth
            depth = rollout_group['depth_images'][frame_idx]  # (H, W) or (H, W, C)
            depth = self._process_depth(depth)

            # Load RGB if needed
            if self.use_rgb:
                rgb = rollout_group['rgb_images'][frame_idx]  # (H, W, 3)
                rgb = self._process_rgb(rgb)
            else:
                rgb = None

            # Load state
            odometry = rollout_group['odometry'][frame_idx]  # Dict-like structure
            state = self._process_state(odometry)

            # Load reference trajectory
            reference = rollout_group['references'][frame_idx]  # (out_seq_len, 3)
            reference = torch.tensor(reference, dtype=torch.float32)

            # Load point cloud if needed
            if self.load_point_clouds and 'point_clouds' in rollout_group:
                point_cloud = rollout_group['point_clouds'][frame_idx]  # (N, 3)
                point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
            else:
                point_cloud = None

            # Current position for collision loss
            current_position = torch.tensor(odometry[:3], dtype=torch.float32)

        # Create sequence dimension
        depth = depth.unsqueeze(0)  # (1, H, W, C)
        state = state.unsqueeze(0)  # (1, state_dim)

        result = {
            'depth': depth,
            'state': state,
            'reference': reference,
            'point_cloud': point_cloud,
            'current_position': current_position
        }

        if rgb is not None:
            result['rgb'] = rgb.unsqueeze(0)

        return result

    def _load_from_directory(self, sample_info: Dict) -> Dict[str, torch.Tensor]:
        """Load sample from directory structure."""
        rollout_dir = sample_info['rollout_dir']
        frame_idx = sample_info['frame_idx']

        # Load depth image
        depth_path = rollout_dir / "depth_images" / f"{frame_idx:06d}.tif"
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        depth = self._process_depth(depth)

        # Load RGB if needed
        if self.use_rgb:
            rgb_path = rollout_dir / "left_images" / f"{frame_idx:06d}.png"
            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = self._process_rgb(rgb)
        else:
            rgb = None

        # Load odometry
        import pandas as pd
        odom_df = pd.read_csv(rollout_dir / "odometry.csv")
        odom_row = odom_df.iloc[frame_idx]
        state = self._process_state_from_csv(odom_row)

        # Load reference trajectory
        ref_df = pd.read_csv(rollout_dir / "expert_trajectories.csv")
        ref_row = ref_df.iloc[frame_idx]
        reference = self._process_reference_from_csv(ref_row)

        # Load point cloud if needed
        if self.load_point_clouds:
            pc_path = rollout_dir / "pointclouds" / f"{frame_idx:06d}.ply"
            if pc_path.exists():
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(str(pc_path))
                point_cloud = torch.tensor(np.asarray(pcd.points), dtype=torch.float32)
            else:
                point_cloud = None
        else:
            point_cloud = None

        # Current position
        current_position = torch.tensor([
            odom_row['pos_x'],
            odom_row['pos_y'],
            odom_row['pos_z']
        ], dtype=torch.float32)

        # Create sequence dimension
        depth = depth.unsqueeze(0)
        state = state.unsqueeze(0)

        result = {
            'depth': depth,
            'state': state,
            'reference': reference,
            'point_cloud': point_cloud,
            'current_position': current_position
        }

        if rgb is not None:
            result['rgb'] = rgb.unsqueeze(0)

        return result

    def _process_depth(self, depth: np.ndarray) -> torch.Tensor:
        """Process depth image."""
        # Resize
        depth_resized = cv2.resize(depth, (self.img_width, self.img_height))

        # Normalize (handle inf values)
        max_depth = np.percentile(depth_resized[depth_resized < np.inf], 99)
        depth_resized = np.clip(depth_resized, 0, max_depth)
        depth_resized = depth_resized / max_depth

        # Convert to 3-channel if needed
        if depth_resized.ndim == 2:
            depth_resized = np.stack([depth_resized] * 3, axis=-1)

        return torch.tensor(depth_resized, dtype=torch.float32)

    def _process_rgb(self, rgb: np.ndarray) -> torch.Tensor:
        """Process RGB image."""
        # Resize
        rgb_resized = cv2.resize(rgb, (self.img_width, self.img_height))

        # Normalize to [0, 1]
        rgb_normalized = rgb_resized.astype(np.float32) / 255.0

        return torch.tensor(rgb_normalized, dtype=torch.float32)

    def _process_state_from_csv(self, odom_row) -> torch.Tensor:
        """Process state from CSV row."""
        features = []

        # Position
        position = np.array([odom_row['pos_x'], odom_row['pos_y'], odom_row['pos_z']])
        if self.use_position:
            features.append(position)

        # Attitude (quaternion to rotation matrix)
        if self.use_attitude:
            quat = np.array([
                odom_row['quat_x'],
                odom_row['quat_y'],
                odom_row['quat_z'],
                odom_row['quat_w']
            ])
            R = Rotation.from_quat(quat).as_matrix()
            features.append(R.flatten())

        # Velocity
        velocity = np.array([odom_row['vel_x'], odom_row['vel_y'], odom_row['vel_z']])
        if self.velocity_frame == 'bf' and self.use_attitude:
            # Transform to body frame
            velocity = R @ velocity
        features.append(velocity)

        # Body rates
        if self.use_bodyrates:
            bodyrates = np.array([
                odom_row['omega_x'],
                odom_row['omega_y'],
                odom_row['omega_z']
            ])
            features.append(bodyrates)

        # Altitude
        features.append(np.array([position[2]]))

        state_vector = np.concatenate(features)
        return torch.tensor(state_vector, dtype=torch.float32)

    def _process_state(self, odom: np.ndarray) -> torch.Tensor:
        """Process state from array (HDF5 format)."""
        # Assuming odom contains: [pos(3), quat(4), vel(3), omega(3)]
        features = []

        position = odom[:3]
        if self.use_position:
            features.append(position)

        if self.use_attitude:
            quat = odom[3:7]
            R = Rotation.from_quat(quat).as_matrix()
            features.append(R.flatten())

        velocity = odom[7:10]
        if self.velocity_frame == 'bf' and self.use_attitude:
            velocity = R @ velocity
        features.append(velocity)

        if self.use_bodyrates:
            bodyrates = odom[10:13]
            features.append(bodyrates)

        features.append(np.array([position[2]]))  # Altitude

        state_vector = np.concatenate(features)
        return torch.tensor(state_vector, dtype=torch.float32)

    def _process_reference_from_csv(self, ref_row) -> torch.Tensor:
        """Process reference trajectory from CSV row."""
        # CSV has columns: timestamp, ref_x0, ref_y0, ref_z0, ..., ref_x9, ref_y9, ref_z9
        reference = []
        for i in range(self.out_seq_len):
            ref_point = [
                ref_row[f'ref_x{i}'],
                ref_row[f'ref_y{i}'],
                ref_row[f'ref_z{i}']
            ]
            reference.append(ref_point)

        reference = np.array(reference, dtype=np.float32)
        return torch.tensor(reference, dtype=torch.float32)


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for DataLoader.

    Handles variable-sized point clouds.
    """
    # Stack regular tensors
    depth = torch.stack([item['depth'] for item in batch])  # (batch, seq_len, H, W, C)
    state = torch.stack([item['state'] for item in batch])  # (batch, seq_len, state_dim)
    reference = torch.stack([item['reference'] for item in batch])  # (batch, out_seq_len, 3)
    current_position = torch.stack([item['current_position'] for item in batch])  # (batch, 3)

    # Point clouds are variable size - keep as list
    point_clouds = [item['point_cloud'] for item in batch]

    # Convert None to empty tensors
    for i, pc in enumerate(point_clouds):
        if pc is None:
            point_clouds[i] = torch.zeros((0, 3), dtype=torch.float32)

    result = {
        'depth': depth,
        'state': state,
        'reference': reference,
        'point_cloud': point_clouds,
        'current_position': current_position
    }

    # Handle optional RGB
    if 'rgb' in batch[0]:
        rgb = torch.stack([item['rgb'] for item in batch])
        result['rgb'] = rgb

    return result


def create_dataloader(
    data_dir: str,
    config: Dict[str, Any],
    mode: str = 'train',
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create DataLoader for training or validation.

    Args:
        data_dir: Data directory
        config: Configuration dictionary
        mode: 'train' or 'val'
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes

    Returns:
        DataLoader
    """
    dataset = RolloutDataset(data_dir, config, mode=mode)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return dataloader
