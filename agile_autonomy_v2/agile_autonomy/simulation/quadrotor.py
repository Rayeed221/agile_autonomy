"""Quadrotor model in PyBullet."""

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from typing import Optional, Tuple

from ..core.state import QuadrotorState


class Quadrotor:
    """
    Quadrotor model using PyBullet physics.

    This creates a simple quadrotor with 4 rotors using PyBullet's
    built-in physics engine.
    """

    def __init__(
        self,
        client_id: int,
        mass: float = 0.775,
        arm_length: float = 0.17,
        start_position: Optional[np.ndarray] = None,
        start_orientation: Optional[Rotation] = None,
        max_thrust: float = 15.0,
        max_torque: float = 0.5
    ):
        """
        Initialize quadrotor.

        Args:
            client_id: PyBullet physics client ID
            mass: Mass in kg
            arm_length: Distance from center to rotor (meters)
            start_position: Initial position [x, y, z]
            start_orientation: Initial orientation (Rotation object)
            max_thrust: Maximum total thrust (Newtons)
            max_torque: Maximum torque per axis (Nm)
        """
        self.client_id = client_id
        self.mass = mass
        self.arm_length = arm_length
        self.max_thrust = max_thrust
        self.max_torque = max_torque

        # Default starting pose
        if start_position is None:
            start_position = np.array([0.0, 0.0, 1.5])
        if start_orientation is None:
            start_orientation = Rotation.identity()

        self.start_position = start_position
        self.start_orientation = start_orientation

        # Create the quadrotor body
        self.body_id = self._create_body()

        # Reset to initial pose
        self.reset()

    def _create_body(self) -> int:
        """Create the quadrotor body in PyBullet."""
        # Create a simple quadrotor using basic shapes
        # Central body (box)
        body_half_extents = [0.05, 0.05, 0.02]

        # Create collision shape
        col_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=body_half_extents,
            physicsClientId=self.client_id
        )

        # Create visual shape
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=body_half_extents,
            rgbaColor=[0.8, 0.1, 0.1, 1.0],
            physicsClientId=self.client_id
        )

        # Create multibody
        body_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=col_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=self.start_position,
            baseOrientation=self._rotation_to_quaternion(self.start_orientation),
            physicsClientId=self.client_id
        )

        # Add visual markers for the rotors (just for visualization)
        rotor_positions = [
            [self.arm_length, 0, 0],
            [-self.arm_length, 0, 0],
            [0, self.arm_length, 0],
            [0, -self.arm_length, 0]
        ]

        for pos in rotor_positions:
            rotor_visual = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=0.05,
                length=0.01,
                rgbaColor=[0.2, 0.2, 0.2, 1.0],
                physicsClientId=self.client_id
            )
            p.createMultiBody(
                baseMass=0,  # Static visual only
                baseVisualShapeIndex=rotor_visual,
                basePosition=np.array(self.start_position) + np.array(pos),
                physicsClientId=self.client_id
            )

        return body_id

    def _rotation_to_quaternion(self, rotation: Rotation) -> list:
        """Convert scipy Rotation to PyBullet quaternion [x, y, z, w]."""
        quat = rotation.as_quat()  # Returns [x, y, z, w]
        return quat.tolist()

    def reset(
        self,
        position: Optional[np.ndarray] = None,
        orientation: Optional[Rotation] = None
    ) -> None:
        """
        Reset quadrotor to a specific pose.

        Args:
            position: Position to reset to (defaults to start_position)
            orientation: Orientation to reset to (defaults to start_orientation)
        """
        if position is None:
            position = self.start_position
        if orientation is None:
            orientation = self.start_orientation

        p.resetBasePositionAndOrientation(
            self.body_id,
            position,
            self._rotation_to_quaternion(orientation),
            physicsClientId=self.client_id
        )

        # Reset velocities to zero
        p.resetBaseVelocity(
            self.body_id,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0],
            physicsClientId=self.client_id
        )

    def get_state(self) -> QuadrotorState:
        """
        Get current state of the quadrotor.

        Returns:
            QuadrotorState object
        """
        # Get position and orientation
        pos, quat = p.getBasePositionAndOrientation(
            self.body_id,
            physicsClientId=self.client_id
        )

        # Get velocities
        lin_vel, ang_vel = p.getBaseVelocity(
            self.body_id,
            physicsClientId=self.client_id
        )

        # Convert to numpy arrays
        position = np.array(pos, dtype=np.float32)
        orientation = Rotation.from_quat(quat)  # PyBullet uses [x,y,z,w]
        velocity = np.array(lin_vel, dtype=np.float32)
        angular_velocity = np.array(ang_vel, dtype=np.float32)

        return QuadrotorState(
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            timestamp=0.0  # Will be set by simulator
        )

    def apply_action(self, thrust: float, torque: np.ndarray) -> None:
        """
        Apply control action to the quadrotor.

        Args:
            thrust: Total thrust force (Newtons) in body z-direction
            torque: Torque vector [tx, ty, tz] in body frame (Nm)
        """
        # Clamp inputs
        thrust = np.clip(thrust, 0, self.max_thrust)
        torque = np.clip(torque, -self.max_torque, self.max_torque)

        # Get current orientation to transform thrust to world frame
        state = self.get_state()
        R = state.rotation_matrix

        # Thrust is in body +z direction
        thrust_body = np.array([0, 0, thrust])
        thrust_world = R.T @ thrust_body

        # Apply force at center of mass (in world frame)
        p.applyExternalForce(
            self.body_id,
            -1,  # Link index (-1 = base)
            thrust_world,
            state.position,
            p.WORLD_FRAME,
            physicsClientId=self.client_id
        )

        # Apply torque (in world frame)
        torque_world = R.T @ torque
        p.applyExternalTorque(
            self.body_id,
            -1,
            torque_world,
            p.WORLD_FRAME,
            physicsClientId=self.client_id
        )

    def check_collision(self) -> bool:
        """
        Check if quadrotor is in collision with environment.

        Returns:
            True if collision detected
        """
        contact_points = p.getContactPoints(
            bodyA=self.body_id,
            physicsClientId=self.client_id
        )
        return len(contact_points) > 0

    def get_collision_force(self) -> float:
        """
        Get total collision force magnitude.

        Returns:
            Total contact force in Newtons
        """
        contact_points = p.getContactPoints(
            bodyA=self.body_id,
            physicsClientId=self.client_id
        )

        total_force = 0.0
        for contact in contact_points:
            total_force += contact[9]  # Normal force

        return total_force

    def remove(self) -> None:
        """Remove the quadrotor from the simulation."""
        p.removeBody(self.body_id, physicsClientId=self.client_id)

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.remove()
        except:
            pass
