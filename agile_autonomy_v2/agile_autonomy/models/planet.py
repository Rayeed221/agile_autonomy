"""PlaNet neural network architecture in PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from typing import Dict, Any, Optional, Tuple
import numpy as np


class PlaNet(nn.Module):
    """
    PlaNet: Planning Network for trajectory prediction.

    Architecture:
    - Image branch: MobileNetV2 backbone + Conv1d processing
    - State branch: Conv1d layers for IMU/state data
    - Plan module: Fuses features and predicts trajectories

    Output: Multi-modal trajectory predictions (3 modes by default)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PlaNet.

        Args:
            config: Configuration dictionary with keys:
                - use_rgb: Use RGB images
                - use_depth: Use depth images
                - img_height: Image height (default 224)
                - img_width: Image width (default 224)
                - state_dim: Output state dimension (default 3 for x,y,z)
                - out_seq_len: Number of future steps to predict (default 10)
                - modes: Number of trajectory modes (default 3)
                - seq_len: Input sequence length (default 1)
                - freeze_backbone: Freeze MobileNet weights
                - inputs: Dict with position, attitude, velocity, bodyrates flags
        """
        super().__init__()

        self.config = config
        self.use_rgb = config.get('use_rgb', False)
        self.use_depth = config.get('use_depth', True)
        self.img_height = config.get('img_height', 224)
        self.img_width = config.get('img_width', 224)
        self.state_dim = config.get('state_dim', 3)
        self.out_seq_len = config.get('out_seq_len', 10)
        self.modes = config.get('modes', 3)
        self.seq_len = config.get('seq_len', 1)
        self.freeze_backbone = config.get('freeze_backbone', False)

        # Input configuration
        inputs_config = config.get('inputs', {})
        self.use_position = inputs_config.get('position', False)
        self.use_attitude = inputs_config.get('attitude', True)
        self.use_velocity = True  # Always use velocity
        self.use_bodyrates = inputs_config.get('bodyrates', True)

        # Calculate state input dimension
        self.state_input_dim = 0
        if self.use_position:
            self.state_input_dim += 3
        if self.use_attitude:
            self.state_input_dim += 9  # Rotation matrix
        self.state_input_dim += 3  # Velocity (always included)
        if self.use_bodyrates:
            self.state_input_dim += 3
        self.state_input_dim += 1  # Altitude

        # Output dimension: positions (x,y,z) * timesteps + alpha (mode weight)
        self.output_dim = self.state_dim * self.out_seq_len + 1

        # Build network
        self._build_network()

    def _build_network(self):
        """Build the network architecture."""

        # Image branch (if using camera)
        if self.use_rgb or self.use_depth:
            # Input channels: 3 for RGB, 3 for depth (as 3-channel), or 6 for both
            in_channels = 3 * int(self.use_rgb) + 3 * int(self.use_depth)

            # MobileNetV2 backbone (pretrained on ImageNet)
            mobilenet = mobilenet_v2(pretrained=True)

            # Modify first conv to accept our input channels if needed
            if in_channels != 3:
                original_conv = mobilenet.features[0][0]
                mobilenet.features[0][0] = nn.Conv2d(
                    in_channels,
                    original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=False
                )

            # Extract features (remove classifier)
            self.backbone = nn.Sequential(*list(mobilenet.features.children()))

            # Freeze backbone if requested
            if self.freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False

            # Global average pooling
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

            # Feature dimension after MobileNet
            backbone_dim = 1280  # MobileNetV2 output

            # Reduce dimension
            self.img_reduce = nn.Conv1d(backbone_dim, 128, kernel_size=1)

            # Image processing layers (Conv1d for temporal)
            self.img_conv = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=2, padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv1d(128, 64, kernel_size=2, padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv1d(64, 64, kernel_size=2, padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv1d(64, 32, kernel_size=2, padding=1),
                nn.LeakyReLU(0.01),
            )

            # Final projection to modes
            self.img_to_modes = nn.Conv1d(32, self.modes, kernel_size=3, padding=1)

            self.has_image_branch = True
        else:
            self.has_image_branch = False

        # State branch (always present)
        self.state_conv = nn.Sequential(
            nn.Conv1d(self.state_input_dim, 64, kernel_size=2, padding=1),
            nn.LeakyReLU(0.5),
            nn.Conv1d(64, 32, kernel_size=2, padding=1),
            nn.LeakyReLU(0.5),
            nn.Conv1d(32, 32, kernel_size=2, padding=1),
            nn.LeakyReLU(0.5),
            nn.Conv1d(32, 32, kernel_size=2, padding=1),
        )

        # State to modes projection
        self.state_to_modes = nn.Conv1d(32, self.modes, kernel_size=3, padding=1)

        # Plan module (fuses image and state features)
        # Input: concatenated features
        if self.has_image_branch:
            plan_input_dim = 64  # 32 from image + 32 from state
        else:
            plan_input_dim = 32  # Only state

        self.plan_module = nn.Sequential(
            nn.Conv1d(plan_input_dim, 64, kernel_size=1),
            nn.LeakyReLU(0.5),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.LeakyReLU(0.5),
            nn.Conv1d(128, 128, kernel_size=1),
            nn.LeakyReLU(0.5),
            nn.Conv1d(128, self.output_dim, kernel_size=1),
        )

    def forward(
        self,
        state: torch.Tensor,
        image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: State tensor (batch, seq_len, state_dim)
            image: Image tensor (batch, seq_len, channels, height, width) or None

        Returns:
            Predictions: (batch, modes, output_dim)
                where output_dim = state_dim * out_seq_len + 1
                Last element in each mode is alpha (mode probability/cost)
        """
        batch_size = state.shape[0]

        # Process state branch
        # state: (batch, seq_len, state_dim) -> (batch, state_dim, seq_len)
        state_permuted = state.permute(0, 2, 1)
        state_features = self.state_conv(state_permuted)  # (batch, 32, seq_len)

        # Project to modes
        state_modes = self.state_to_modes(state_features)  # (batch, modes, seq_len)

        # Average over sequence dimension
        state_modes = state_modes.mean(dim=2, keepdim=True)  # (batch, modes, 1)
        state_modes = state_modes.squeeze(2)  # (batch, modes)
        state_modes = state_modes.unsqueeze(2)  # (batch, modes, 1) for concatenation

        # Reshape state features for concatenation
        state_features = state_features.mean(dim=2)  # (batch, 32)
        state_features = state_features.unsqueeze(1).expand(-1, self.modes, -1)  # (batch, modes, 32)

        # Process image branch if present
        if self.has_image_branch and image is not None:
            # image: (batch, seq_len, channels, height, width)
            # Process each timestep through backbone
            seq_features = []
            for t in range(self.seq_len):
                img_t = image[:, t, :, :, :]  # (batch, channels, H, W)
                feat_t = self.backbone(img_t)  # (batch, 1280, H', W')
                feat_t = self.global_pool(feat_t)  # (batch, 1280, 1, 1)
                feat_t = feat_t.squeeze(-1).squeeze(-1)  # (batch, 1280)
                seq_features.append(feat_t)

            # Stack sequence
            img_features = torch.stack(seq_features, dim=2)  # (batch, 1280, seq_len)

            # Reduce dimension
            img_features = self.img_reduce(img_features)  # (batch, 128, seq_len)

            # Process through conv layers
            img_features = self.img_conv(img_features)  # (batch, 32, seq_len)

            # Project to modes
            img_modes = self.img_to_modes(img_features)  # (batch, modes, seq_len)
            img_modes = img_modes.mean(dim=2, keepdim=True)  # (batch, modes, 1)
            img_modes = img_modes.squeeze(2)  # (batch, modes)
            img_modes = img_modes.unsqueeze(2)  # (batch, modes, 1)

            # Reshape for concatenation
            img_features = img_features.mean(dim=2)  # (batch, 32)
            img_features = img_features.unsqueeze(1).expand(-1, self.modes, -1)  # (batch, modes, 32)

            # Concatenate image and state features
            fused_features = torch.cat([img_features, state_features], dim=2)  # (batch, modes, 64)
        else:
            fused_features = state_features  # (batch, modes, 32)

        # Reshape for Conv1d: (batch * modes, features, 1)
        fused_features = fused_features.reshape(batch_size * self.modes, -1, 1)

        # Plan module
        predictions = self.plan_module(fused_features)  # (batch * modes, output_dim, 1)
        predictions = predictions.squeeze(2)  # (batch * modes, output_dim)

        # Reshape to (batch, modes, output_dim)
        predictions = predictions.reshape(batch_size, self.modes, self.output_dim)

        return predictions

    def predict_trajectories(
        self,
        state: torch.Tensor,
        image: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict trajectories and mode weights.

        Args:
            state: State tensor
            image: Image tensor or None

        Returns:
            Tuple of (trajectories, alphas)
            - trajectories: (batch, modes, out_seq_len, state_dim)
            - alphas: (batch, modes) - mode selection weights
        """
        predictions = self.forward(state, image)  # (batch, modes, output_dim)

        # Split into trajectories and alphas
        trajectories = predictions[:, :, :-1]  # (batch, modes, state_dim * out_seq_len)
        alphas = predictions[:, :, -1]  # (batch, modes)

        # Reshape trajectories
        batch_size = predictions.shape[0]
        trajectories = trajectories.reshape(
            batch_size, self.modes, self.out_seq_len, self.state_dim
        )

        return trajectories, alphas

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config


def create_model(config: Dict[str, Any]) -> PlaNet:
    """
    Create PlaNet model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        PlaNet model
    """
    return PlaNet(config)
