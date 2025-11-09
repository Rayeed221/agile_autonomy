"""Quadrotor state representation."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class QuadrotorState:
    """
    Complete state representation of a quadrotor.

    Attributes:
        position: 3D position in world frame [x, y, z] (meters)
        velocity: 3D velocity in world frame [vx, vy, vz] (m/s)
        orientation: Rotation object representing attitude (world to body)
        angular_velocity: Angular velocity in body frame [wx, wy, wz] (rad/s)
        timestamp: Time of this state (seconds)
    """

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: Rotation = field(default_factory=lambda: Rotation.identity())
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    timestamp: float = 0.0

    def __post_init__(self):
        """Validate and convert types."""
        self.position = np.asarray(self.position, dtype=np.float32)
        self.velocity = np.asarray(self.velocity, dtype=np.float32)
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float32)

        if not isinstance(self.orientation, Rotation):
            raise TypeError("orientation must be a scipy.spatial.transform.Rotation object")

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get rotation matrix (3x3) from world to body frame."""
        return self.orientation.as_matrix()

    @property
    def quaternion(self) -> np.ndarray:
        """Get quaternion [w, x, y, z]."""
        quat = self.orientation.as_quat()  # Returns [x, y, z, w]
        return np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z]

    @property
    def euler_angles(self) -> np.ndarray:
        """Get Euler angles [roll, pitch, yaw] in radians."""
        return self.orientation.as_euler('xyz', degrees=False)

    @property
    def velocity_body_frame(self) -> np.ndarray:
        """Get velocity in body frame."""
        return self.rotation_matrix @ self.velocity

    @property
    def altitude(self) -> float:
        """Get altitude (z position)."""
        return self.position[2]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'quaternion': self.quaternion.tolist(),
            'angular_velocity': self.angular_velocity.tolist(),
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'QuadrotorState':
        """Create from dictionary."""
        quat = np.array(data['quaternion'])  # [w, x, y, z]
        # Convert to scipy format [x, y, z, w]
        quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])

        return cls(
            position=np.array(data['position']),
            velocity=np.array(data['velocity']),
            orientation=Rotation.from_quat(quat_scipy),
            angular_velocity=np.array(data['angular_velocity']),
            timestamp=data['timestamp']
        )

    def to_network_input(
        self,
        include_position: bool = False,
        include_attitude: bool = True,
        include_velocity: bool = True,
        include_bodyrates: bool = True,
        velocity_frame: str = 'body'
    ) -> np.ndarray:
        """
        Convert state to network input format.

        Args:
            include_position: Include 3D position
            include_attitude: Include rotation matrix (9 values)
            include_velocity: Include velocity (3 values)
            include_bodyrates: Include angular velocity (3 values)
            velocity_frame: 'body' or 'world'

        Returns:
            State vector as numpy array
        """
        features = []

        if include_position:
            features.append(self.position)

        if include_attitude:
            features.append(self.rotation_matrix.flatten())

        if include_velocity:
            if velocity_frame == 'body':
                features.append(self.velocity_body_frame)
            else:
                features.append(self.velocity)

        if include_bodyrates:
            features.append(self.angular_velocity)

        # Add altitude as separate feature
        features.append(np.array([self.altitude]))

        return np.concatenate(features).astype(np.float32)

    def copy(self) -> 'QuadrotorState':
        """Create a deep copy of this state."""
        return QuadrotorState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=Rotation.from_matrix(self.rotation_matrix.copy()),
            angular_velocity=self.angular_velocity.copy(),
            timestamp=self.timestamp
        )

    def __repr__(self) -> str:
        return (
            f"QuadrotorState(\n"
            f"  pos={self.position}, \n"
            f"  vel={self.velocity}, \n"
            f"  euler={np.rad2deg(self.euler_angles):.1f}°, \n"
            f"  t={self.timestamp:.3f}s\n"
            f")"
        )
