"""Trajectory representation and utilities."""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class TrajectoryPoint:
    """Single point in a trajectory."""

    position: np.ndarray  # 3D position [x, y, z]
    velocity: Optional[np.ndarray] = None  # 3D velocity [vx, vy, vz]
    acceleration: Optional[np.ndarray] = None  # 3D acceleration
    yaw: float = 0.0  # Yaw angle in radians
    timestamp: float = 0.0  # Time offset from trajectory start

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float32)
        if self.velocity is not None:
            self.velocity = np.asarray(self.velocity, dtype=np.float32)
        if self.acceleration is not None:
            self.acceleration = np.asarray(self.acceleration, dtype=np.float32)


class Trajectory:
    """
    Trajectory representation for quadrotor motion.

    A trajectory is a sequence of waypoints with timestamps.
    Supports both body-frame and world-frame representations.
    """

    def __init__(
        self,
        points: Optional[List[TrajectoryPoint]] = None,
        frame: str = 'world',
        dt: float = 0.1
    ):
        """
        Initialize trajectory.

        Args:
            points: List of trajectory points
            frame: 'world' or 'body' frame
            dt: Time step between points (seconds)
        """
        self.points = points if points is not None else []
        self.frame = frame
        self.dt = dt

    @classmethod
    def from_positions(
        cls,
        positions: np.ndarray,
        frame: str = 'world',
        dt: float = 0.1
    ) -> 'Trajectory':
        """
        Create trajectory from position array.

        Args:
            positions: Array of shape (N, 3) with N waypoints
            frame: 'world' or 'body'
            dt: Time step between points

        Returns:
            Trajectory object
        """
        positions = np.asarray(positions)
        if positions.ndim == 1:
            positions = positions.reshape(-1, 3)

        points = []
        for i, pos in enumerate(positions):
            points.append(TrajectoryPoint(
                position=pos,
                timestamp=i * dt
            ))

        return cls(points=points, frame=frame, dt=dt)

    @classmethod
    def from_network_output(
        cls,
        output: np.ndarray,
        dt: float = 0.1,
        frame: str = 'body'
    ) -> 'Trajectory':
        """
        Create trajectory from neural network output.

        Args:
            output: Array of shape (N,) or (N, 3) where N = num_steps * 3
            dt: Time step between points
            frame: Reference frame

        Returns:
            Trajectory object
        """
        output = np.asarray(output)

        # Reshape if flat array
        if output.ndim == 1:
            output = output.reshape(-1, 3)

        return cls.from_positions(output, frame=frame, dt=dt)

    def to_positions(self) -> np.ndarray:
        """Get positions as numpy array of shape (N, 3)."""
        return np.array([p.position for p in self.points], dtype=np.float32)

    def to_velocities(self) -> Optional[np.ndarray]:
        """Get velocities as numpy array of shape (N, 3), or None if not available."""
        if not self.points or self.points[0].velocity is None:
            return None
        return np.array([p.velocity for p in self.points], dtype=np.float32)

    def transform_to_world_frame(
        self,
        current_position: np.ndarray,
        current_rotation_matrix: np.ndarray
    ) -> 'Trajectory':
        """
        Transform trajectory from body frame to world frame.

        Args:
            current_position: Current position in world frame (3,)
            current_rotation_matrix: Current rotation matrix (3, 3)

        Returns:
            New trajectory in world frame
        """
        if self.frame == 'world':
            return self

        world_points = []
        for point in self.points:
            # Transform position
            world_pos = current_position + current_rotation_matrix.T @ point.position

            # Transform velocity if available
            world_vel = None
            if point.velocity is not None:
                world_vel = current_rotation_matrix.T @ point.velocity

            # Transform acceleration if available
            world_acc = None
            if point.acceleration is not None:
                world_acc = current_rotation_matrix.T @ point.acceleration

            world_points.append(TrajectoryPoint(
                position=world_pos,
                velocity=world_vel,
                acceleration=world_acc,
                yaw=point.yaw,
                timestamp=point.timestamp
            ))

        return Trajectory(points=world_points, frame='world', dt=self.dt)

    def compute_velocities(self) -> None:
        """Compute velocities from positions using finite differences."""
        if len(self.points) < 2:
            return

        positions = self.to_positions()
        velocities = np.diff(positions, axis=0) / self.dt

        # Set velocities (forward difference for last point)
        for i in range(len(velocities)):
            self.points[i].velocity = velocities[i]

        # Last point uses same velocity as second-to-last
        self.points[-1].velocity = velocities[-1]

    def compute_accelerations(self) -> None:
        """Compute accelerations from velocities using finite differences."""
        if len(self.points) < 2:
            return

        # First compute velocities if not available
        if self.points[0].velocity is None:
            self.compute_velocities()

        velocities = self.to_velocities()
        if velocities is None:
            return

        accelerations = np.diff(velocities, axis=0) / self.dt

        # Set accelerations
        for i in range(len(accelerations)):
            self.points[i].acceleration = accelerations[i]

        # Last point uses same acceleration
        self.points[-1].acceleration = accelerations[-1]

    def get_point_at_time(self, t: float) -> Optional[TrajectoryPoint]:
        """
        Get interpolated point at specific time.

        Args:
            t: Time from trajectory start

        Returns:
            Interpolated trajectory point, or None if out of range
        """
        if not self.points or t < 0:
            return None

        # Find surrounding points
        idx = int(t / self.dt)
        if idx >= len(self.points) - 1:
            return self.points[-1]

        # Linear interpolation
        alpha = (t - self.points[idx].timestamp) / self.dt
        p1, p2 = self.points[idx], self.points[idx + 1]

        interp_pos = (1 - alpha) * p1.position + alpha * p2.position
        interp_vel = None
        if p1.velocity is not None and p2.velocity is not None:
            interp_vel = (1 - alpha) * p1.velocity + alpha * p2.velocity

        return TrajectoryPoint(
            position=interp_pos,
            velocity=interp_vel,
            yaw=(1 - alpha) * p1.yaw + alpha * p2.yaw,
            timestamp=t
        )

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> TrajectoryPoint:
        return self.points[idx]

    def __repr__(self) -> str:
        return f"Trajectory(points={len(self.points)}, frame={self.frame}, dt={self.dt})"
