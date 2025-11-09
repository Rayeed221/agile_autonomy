"""Sensor simulation for quadrotor."""

import numpy as np
import pybullet as p
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation

from ..core.state import QuadrotorState


class Camera:
    """
    RGB camera sensor using PyBullet rendering.
    """

    def __init__(
        self,
        client_id: int,
        width: int = 640,
        height: int = 480,
        fov: float = 90.0,
        near: float = 0.1,
        far: float = 100.0,
        camera_offset: Optional[np.ndarray] = None,
        camera_rotation: Optional[Rotation] = None
    ):
        """
        Initialize camera.

        Args:
            client_id: PyBullet client ID
            width: Image width (pixels)
            height: Image height (pixels)
            fov: Horizontal field of view (degrees)
            near: Near clipping plane (meters)
            far: Far clipping plane (meters)
            camera_offset: Offset from quadrotor center in body frame
            camera_rotation: Camera rotation relative to body frame
        """
        self.client_id = client_id
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far

        # Camera mounting (default: pointing forward, slightly down)
        if camera_offset is None:
            camera_offset = np.array([0.1, 0.0, 0.0])  # 10cm forward
        if camera_rotation is None:
            # Pitch down by 0 degrees (level)
            camera_rotation = Rotation.identity()

        self.camera_offset = camera_offset
        self.camera_rotation = camera_rotation

        # Compute projection matrix
        aspect = width / height
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=near,
            farVal=far
        )

    def capture(self, quad_state: QuadrotorState) -> np.ndarray:
        """
        Capture RGB image from quadrotor's perspective.

        Args:
            quad_state: Current quadrotor state

        Returns:
            RGB image as numpy array of shape (height, width, 3), dtype uint8
        """
        # Compute camera pose in world frame
        camera_pos, camera_target, up_vector = self._compute_camera_pose(quad_state)

        # Compute view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=camera_target,
            cameraUpVector=up_vector,
            physicsClientId=self.client_id
        )

        # Render
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            physicsClientId=self.client_id
        )

        # Extract RGB (discard alpha channel)
        rgb_array = np.array(rgb_img, dtype=np.uint8).reshape(self.height, self.width, 4)
        rgb_array = rgb_array[:, :, :3]  # Remove alpha

        return rgb_array

    def _compute_camera_pose(
        self,
        quad_state: QuadrotorState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute camera position, target, and up vector in world frame.

        Returns:
            Tuple of (camera_pos, camera_target, up_vector)
        """
        # Transform camera offset to world frame
        R_world_body = quad_state.rotation_matrix.T  # Body to world
        camera_pos_world = quad_state.position + R_world_body @ self.camera_offset

        # Camera looks along its local +X axis after applying camera_rotation
        # Camera up is local +Z
        R_camera = (Rotation.from_matrix(R_world_body) * self.camera_rotation).as_matrix()

        # Forward direction (camera looks along +X in camera frame)
        camera_forward = R_camera @ np.array([1.0, 0.0, 0.0])
        camera_up = R_camera @ np.array([0.0, 0.0, 1.0])

        # Target is 1 meter ahead
        camera_target = camera_pos_world + camera_forward

        return camera_pos_world, camera_target, camera_up


class DepthCamera(Camera):
    """
    Depth camera sensor using PyBullet rendering.

    Extends RGB camera to also provide depth information.
    """

    def capture_depth(self, quad_state: QuadrotorState) -> np.ndarray:
        """
        Capture depth image.

        Args:
            quad_state: Current quadrotor state

        Returns:
            Depth image as numpy array of shape (height, width), dtype float32
            Values in meters, inf for invalid/far pixels
        """
        # Compute camera pose
        camera_pos, camera_target, up_vector = self._compute_camera_pose(quad_state)

        # Compute view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=camera_target,
            cameraUpVector=up_vector,
            physicsClientId=self.client_id
        )

        # Render
        width, height, rgb_img, depth_buffer, seg_img = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            physicsClientId=self.client_id
        )

        # Convert depth buffer to actual depth values
        depth_buffer = np.array(depth_buffer, dtype=np.float32).reshape(self.height, self.width)

        # PyBullet returns depth in range [0, 1] which needs to be converted
        # depth_meters = far * near / (far - (far - near) * depth_buffer)
        depth_meters = self.far * self.near / (self.far - (self.far - self.near) * depth_buffer)

        # Set very far values to inf
        depth_meters[depth_meters > self.far * 0.99] = np.inf

        return depth_meters

    def capture_rgbd(self, quad_state: QuadrotorState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture both RGB and depth.

        Args:
            quad_state: Current quadrotor state

        Returns:
            Tuple of (rgb_image, depth_image)
        """
        rgb = self.capture(quad_state)
        depth = self.capture_depth(quad_state)
        return rgb, depth


class IMU:
    """
    Inertial Measurement Unit sensor.

    Provides accelerometer and gyroscope readings.
    In simulation, we can get perfect measurements from PyBullet.
    Optional noise can be added for realism.
    """

    def __init__(
        self,
        accel_noise_std: float = 0.0,
        gyro_noise_std: float = 0.0,
        gravity: float = 9.81
    ):
        """
        Initialize IMU.

        Args:
            accel_noise_std: Standard deviation of accelerometer noise (m/s^2)
            gyro_noise_std: Standard deviation of gyroscope noise (rad/s)
            gravity: Gravity magnitude (m/s^2)
        """
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.gravity = gravity

    def measure(self, quad_state: QuadrotorState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get IMU measurements.

        Args:
            quad_state: Current quadrotor state

        Returns:
            Tuple of (linear_acceleration, angular_velocity)
            - linear_acceleration: 3D acceleration in body frame (m/s^2)
            - angular_velocity: 3D angular velocity in body frame (rad/s)
        """
        # In a real IMU, we'd need to compute acceleration from velocity changes
        # For now, we'll use the body rates directly (perfect measurement)

        # Angular velocity (already in body frame)
        gyro = quad_state.angular_velocity.copy()

        # Add noise
        if self.gyro_noise_std > 0:
            gyro += np.random.normal(0, self.gyro_noise_std, size=3)

        # For linear acceleration, we need to:
        # 1. Account for gravity
        # 2. Transform to body frame
        # In simulation, we approximate this (proper implementation would track velocity changes)

        R = quad_state.rotation_matrix
        gravity_world = np.array([0, 0, -self.gravity])
        gravity_body = R @ gravity_world

        # Approximate specific force (would need velocity derivative for full accuracy)
        accel = gravity_body.copy()

        # Add noise
        if self.accel_noise_std > 0:
            accel += np.random.normal(0, self.accel_noise_std, size=3)

        return accel, gyro
