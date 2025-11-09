"""Loss functions for training PlaNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import numpy as np
from scipy.spatial import cKDTree


class MixtureSpaceLoss(nn.Module):
    """
    Mixture of experts loss for multi-modal trajectory prediction.

    This loss computes the weighted MSE between predicted trajectories
    and ground truth, using the predicted mode weights (alphas).
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Initialize MixtureSpaceLoss.

        Args:
            reduction: 'mean' or 'sum' or 'none'
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mixture space loss.

        Args:
            predictions: (batch, modes, output_dim) where last element is alpha
            targets: (batch, output_dim-1) ground truth trajectories

        Returns:
            Loss scalar
        """
        batch_size, modes, output_dim = predictions.shape

        # Split predictions into trajectories and alphas
        pred_trajectories = predictions[:, :, :-1]  # (batch, modes, traj_dim)
        alphas = predictions[:, :, -1]  # (batch, modes)

        # Expand targets for each mode
        targets_expanded = targets.unsqueeze(1).expand(-1, modes, -1)  # (batch, modes, traj_dim)

        # Compute MSE for each mode
        mse_per_mode = F.mse_loss(
            pred_trajectories,
            targets_expanded,
            reduction='none'
        )  # (batch, modes, traj_dim)

        # Average over trajectory dimension
        mse_per_mode = mse_per_mode.mean(dim=2)  # (batch, modes)

        # Weight by alpha (soft selection)
        # Use softmax to normalize alphas
        weights = F.softmax(-alphas, dim=1)  # Lower alpha = higher weight

        # Weighted loss
        weighted_loss = (weights * mse_per_mode).sum(dim=1)  # (batch,)

        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class TrajectoryCostLoss(nn.Module):
    """
    Collision-aware trajectory cost loss.

    Evaluates predicted trajectories against 3D point clouds to penalize
    collisions. Uses KD-tree for efficient nearest neighbor search.
    """

    def __init__(
        self,
        quadrotor_radius: float = 0.3,
        environment_threshold: float = 0.8,
        cost_scale: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize TrajectoryCostLoss.

        Args:
            quadrotor_radius: Collision radius of quadrotor (meters)
            environment_threshold: Safety distance from obstacles (meters)
            cost_scale: Scale factor for loss
            reduction: 'mean' or 'sum' or 'none'
        """
        super().__init__()
        self.quadrotor_radius = quadrotor_radius
        self.environment_threshold = environment_threshold
        self.cost_scale = cost_scale
        self.reduction = reduction

    def compute_trajectory_cost(
        self,
        trajectory: np.ndarray,
        point_cloud: np.ndarray
    ) -> float:
        """
        Compute collision cost for a single trajectory.

        Args:
            trajectory: (num_steps, 3) trajectory positions
            point_cloud: (N, 3) environment point cloud

        Returns:
            Collision cost (0 = no collision, higher = more collisions)
        """
        if len(point_cloud) == 0:
            return 0.0

        # Build KD-tree for fast nearest neighbor search
        tree = cKDTree(point_cloud)

        # Query distances to nearest obstacles for each waypoint
        distances, _ = tree.query(trajectory, k=1)

        # Compute cost (distance below threshold)
        threshold = self.quadrotor_radius + self.environment_threshold
        violations = np.maximum(0, threshold - distances)

        # Sum violations
        cost = violations.sum()

        return cost

    def forward(
        self,
        predictions: torch.Tensor,
        point_clouds: list,
        current_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute trajectory cost loss.

        Args:
            predictions: (batch, modes, output_dim) predicted trajectories + alphas
            point_clouds: List of numpy arrays, each (N_i, 3) for batch item i
            current_positions: (batch, 3) current positions for trajectory transformation

        Returns:
            Loss scalar
        """
        batch_size, modes, output_dim = predictions.shape
        device = predictions.device

        # Split predictions
        pred_trajectories = predictions[:, :, :-1]  # (batch, modes, traj_dim)
        pred_alphas = predictions[:, :, -1]  # (batch, modes)

        # Reshape trajectories: (batch, modes, num_steps, 3)
        num_steps = (output_dim - 1) // 3
        pred_trajectories = pred_trajectories.reshape(batch_size, modes, num_steps, 3)

        # Compute costs for each trajectory
        costs = torch.zeros(batch_size, modes, device=device)

        for b in range(batch_size):
            if point_clouds[b] is None or len(point_clouds[b]) == 0:
                continue

            for m in range(modes):
                # Get trajectory in world frame
                # Assuming predictions are in body frame, transform to world
                # For simplicity, we'll add current position (simplified transformation)
                traj = pred_trajectories[b, m].detach().cpu().numpy()  # (num_steps, 3)

                # Add current position (simplified - should use full rotation)
                current_pos = current_positions[b].detach().cpu().numpy()
                traj_world = traj + current_pos

                # Compute cost
                cost = self.compute_trajectory_cost(traj_world, point_clouds[b])
                costs[b, m] = cost

        # Normalize costs
        costs = costs / (num_steps + 1e-6)

        # Loss: MSE between predicted alphas and computed costs
        # Alphas should predict collision risk
        loss = F.mse_loss(pred_alphas, costs, reduction='none')  # (batch, modes)

        # Average over modes
        loss = loss.mean(dim=1)  # (batch,)

        # Apply scale
        loss = self.cost_scale * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combined loss: MixtureSpaceLoss + TrajectoryCostLoss.

    This is the main loss used for training.
    """

    def __init__(
        self,
        mixture_weight: float = 1.0,
        collision_weight: float = 1.0,
        quadrotor_radius: float = 0.3,
        environment_threshold: float = 0.8
    ):
        """
        Initialize CombinedLoss.

        Args:
            mixture_weight: Weight for mixture space loss
            collision_weight: Weight for trajectory cost loss
            quadrotor_radius: Collision radius
            environment_threshold: Safety distance
        """
        super().__init__()

        self.mixture_weight = mixture_weight
        self.collision_weight = collision_weight

        self.mixture_loss = MixtureSpaceLoss()
        self.collision_loss = TrajectoryCostLoss(
            quadrotor_radius=quadrotor_radius,
            environment_threshold=environment_threshold
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        point_clouds: Optional[list] = None,
        current_positions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            predictions: (batch, modes, output_dim)
            targets: (batch, output_dim-1)
            point_clouds: List of point clouds (optional)
            current_positions: (batch, 3) current positions (optional)

        Returns:
            Dictionary with 'total', 'mixture', 'collision' losses
        """
        # Mixture space loss
        mixture_loss = self.mixture_weight * self.mixture_loss(predictions, targets)

        # Trajectory cost loss (if point clouds provided)
        if point_clouds is not None and current_positions is not None:
            collision_loss = self.collision_weight * self.collision_loss(
                predictions, point_clouds, current_positions
            )
        else:
            collision_loss = torch.tensor(0.0, device=predictions.device)

        # Total loss
        total_loss = mixture_loss + collision_loss

        return {
            'total': total_loss,
            'mixture': mixture_loss,
            'collision': collision_loss
        }


# Alias for backward compatibility
class SpaceLoss(MixtureSpaceLoss):
    """Alias for MixtureSpaceLoss."""
    pass
