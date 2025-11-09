"""Controllers for quadrotor stabilization and trajectory tracking."""

import numpy as np
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation

from ..core.state import QuadrotorState
from ..core.trajectory import Trajectory, TrajectoryPoint


class PIDController:
    """
    PID controller for position and attitude control of quadrotor.

    This implements a cascaded control architecture:
    - Outer loop: Position control (generates desired thrust and attitude)
    - Inner loop: Attitude control (generates torques)
    """

    def __init__(
        self,
        mass: float = 0.775,
        gravity: float = 9.81,
        # Position gains
        kp_pos: float = 1.0,
        kd_pos: float = 0.5,
        ki_pos: float = 0.0,
        # Attitude gains
        kp_att: float = 3.0,
        kd_att: float = 0.3,
        ki_att: float = 0.0,
        # Limits
        max_tilt: float = np.pi / 4,  # 45 degrees
        max_thrust: float = 15.0,
        max_torque: float = 0.5
    ):
        """
        Initialize PID controller.

        Args:
            mass: Quadrotor mass (kg)
            gravity: Gravity (m/s^2)
            kp_pos: Position proportional gain
            kd_pos: Position derivative gain
            ki_pos: Position integral gain
            kp_att: Attitude proportional gain
            kd_att: Attitude derivative gain
            ki_att: Attitude integral gain
            max_tilt: Maximum tilt angle (radians)
            max_thrust: Maximum thrust (N)
            max_torque: Maximum torque per axis (Nm)
        """
        self.mass = mass
        self.gravity = gravity

        # Position PID gains
        self.kp_pos = np.array([kp_pos, kp_pos, kp_pos])
        self.kd_pos = np.array([kd_pos, kd_pos, kd_pos])
        self.ki_pos = np.array([ki_pos, ki_pos, ki_pos])

        # Attitude PID gains
        self.kp_att = np.array([kp_att, kp_att, kp_att])
        self.kd_att = np.array([kd_att, kd_att, kd_att])
        self.ki_att = np.array([ki_att, ki_att, ki_att])

        # Limits
        self.max_tilt = max_tilt
        self.max_thrust = max_thrust
        self.max_torque = max_torque

        # Integral error accumulators
        self.pos_error_integral = np.zeros(3)
        self.att_error_integral = np.zeros(3)

        # Last update time
        self.last_time = None

    def reset(self) -> None:
        """Reset controller state (integral errors)."""
        self.pos_error_integral = np.zeros(3)
        self.att_error_integral = np.zeros(3)
        self.last_time = None

    def compute_control(
        self,
        state: QuadrotorState,
        target_position: np.ndarray,
        target_velocity: Optional[np.ndarray] = None,
        target_yaw: float = 0.0,
        dt: float = 0.01
    ) -> Tuple[float, np.ndarray]:
        """
        Compute control action for position tracking.

        Args:
            state: Current quadrotor state
            target_position: Desired position [x, y, z]
            target_velocity: Desired velocity [vx, vy, vz] (optional)
            target_yaw: Desired yaw angle (radians)
            dt: Time step (seconds)

        Returns:
            Tuple of (thrust, torque)
            - thrust: Total thrust force (N)
            - torque: Torque vector [tx, ty, tz] in body frame (Nm)
        """
        if target_velocity is None:
            target_velocity = np.zeros(3)

        # Position control (outer loop)
        position_error = target_position - state.position
        velocity_error = target_velocity - state.velocity

        # Update integral
        self.pos_error_integral += position_error * dt

        # PID for position
        desired_acceleration = (
            self.kp_pos * position_error +
            self.kd_pos * velocity_error +
            self.ki_pos * self.pos_error_integral
        )

        # Add gravity compensation
        desired_acceleration[2] += self.gravity

        # Compute desired thrust magnitude and direction
        thrust_magnitude = self.mass * np.linalg.norm(desired_acceleration)
        thrust_magnitude = np.clip(thrust_magnitude, 0, self.max_thrust)

        # Desired thrust direction (in world frame)
        if np.linalg.norm(desired_acceleration) > 1e-6:
            thrust_direction = desired_acceleration / np.linalg.norm(desired_acceleration)
        else:
            thrust_direction = np.array([0, 0, 1])

        # Compute desired attitude from thrust direction
        # We want the body z-axis to align with thrust direction
        desired_z_world = thrust_direction

        # Constrain tilt
        tilt_angle = np.arccos(np.clip(desired_z_world[2], -1, 1))
        if tilt_angle > self.max_tilt:
            # Limit tilt by rotating towards vertical
            vertical = np.array([0, 0, 1])
            axis = np.cross(vertical, desired_z_world)
            if np.linalg.norm(axis) > 1e-6:
                axis = axis / np.linalg.norm(axis)
                desired_z_world = Rotation.from_rotvec(
                    axis * self.max_tilt
                ).apply(vertical)

        # Construct desired rotation matrix
        # z-axis is thrust direction
        # x-axis in horizontal plane pointing towards yaw direction
        yaw_vec = np.array([np.cos(target_yaw), np.sin(target_yaw), 0])
        desired_x_world = np.cross(yaw_vec, desired_z_world)
        if np.linalg.norm(desired_x_world) > 1e-6:
            desired_x_world = desired_x_world / np.linalg.norm(desired_x_world)
        else:
            desired_x_world = np.array([1, 0, 0])

        desired_y_world = np.cross(desired_z_world, desired_x_world)

        desired_R_world_body = np.column_stack([desired_x_world, desired_y_world, desired_z_world])

        # Attitude control (inner loop)
        current_R = state.rotation_matrix.T  # Body to world

        # Rotation error (simplified - should use proper SO(3) error)
        R_error = desired_R_world_body.T @ current_R
        rotation_error_matrix = (R_error - R_error.T) / 2

        # Convert to axis-angle (approximate)
        attitude_error = np.array([
            rotation_error_matrix[2, 1],
            rotation_error_matrix[0, 2],
            rotation_error_matrix[1, 0]
        ])

        # Angular velocity error
        angular_vel_error = -state.angular_velocity  # Desired omega is zero for hover

        # Update integral
        self.att_error_integral += attitude_error * dt

        # PID for attitude (produces torque in body frame)
        torque = (
            self.kp_att * attitude_error +
            self.kd_att * angular_vel_error +
            self.ki_att * self.att_error_integral
        )

        # Clip torque
        torque = np.clip(torque, -self.max_torque, self.max_torque)

        return thrust_magnitude, torque

    def compute_trajectory_control(
        self,
        state: QuadrotorState,
        trajectory: Trajectory,
        time_in_trajectory: float,
        dt: float = 0.01
    ) -> Tuple[float, np.ndarray]:
        """
        Compute control for trajectory tracking.

        Args:
            state: Current state
            trajectory: Trajectory to track
            time_in_trajectory: Time since trajectory start
            dt: Time step

        Returns:
            Tuple of (thrust, torque)
        """
        # Get target point from trajectory
        target_point = trajectory.get_point_at_time(time_in_trajectory)

        if target_point is None:
            # Trajectory ended, hover at last position
            target_point = trajectory.points[-1]

        # Transform to world frame if needed
        if trajectory.frame == 'body':
            # This is a simplified version - proper implementation would
            # transform the entire trajectory at each timestep
            target_position = state.position + state.rotation_matrix.T @ target_point.position
        else:
            target_position = target_point.position

        target_velocity = target_point.velocity if target_point.velocity is not None else None
        target_yaw = target_point.yaw

        return self.compute_control(
            state=state,
            target_position=target_position,
            target_velocity=target_velocity,
            target_yaw=target_yaw,
            dt=dt
        )


class SimpleHoverController(PIDController):
    """Simplified controller for basic hovering at a setpoint."""

    def __init__(self, hover_height: float = 1.5, **kwargs):
        """
        Initialize hover controller.

        Args:
            hover_height: Desired hover height (meters)
            **kwargs: Additional arguments passed to PIDController
        """
        super().__init__(**kwargs)
        self.hover_height = hover_height

    def compute_hover_control(
        self,
        state: QuadrotorState,
        dt: float = 0.01
    ) -> Tuple[float, np.ndarray]:
        """
        Compute control to hover at fixed height.

        Args:
            state: Current state
            dt: Time step

        Returns:
            Tuple of (thrust, torque)
        """
        hover_position = np.array([
            state.position[0],  # Stay at current x
            state.position[1],  # Stay at current y
            self.hover_height   # Fixed height
        ])

        return self.compute_control(
            state=state,
            target_position=hover_position,
            target_velocity=np.zeros(3),
            target_yaw=0.0,
            dt=dt
        )
