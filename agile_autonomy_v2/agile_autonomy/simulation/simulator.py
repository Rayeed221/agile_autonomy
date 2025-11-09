"""Main quadrotor simulator integrating all components."""

import numpy as np
import pybullet as p
import pybullet_data
from typing import Optional, Dict, Any, Tuple
from scipy.spatial.transform import Rotation
import time

from ..core.state import QuadrotorState
from ..core.trajectory import Trajectory
from ..core.config import Config
from .quadrotor import Quadrotor
from .environment import Environment, ForestEnvironment
from .sensors import Camera, DepthCamera, IMU
from .controller import PIDController


class QuadrotorSimulator:
    """
    Main simulator class for quadrotor navigation.

    Integrates:
    - PyBullet physics
    - Quadrotor model
    - Environment
    - Sensors (camera, depth, IMU)
    - Controller
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        environment: Optional[Environment] = None,
        gui: bool = True,
        physics_timestep: float = 1.0/240.0
    ):
        """
        Initialize simulator.

        Args:
            config: Configuration object (uses defaults if None)
            environment: Environment object (creates forest if None)
            gui: Whether to show GUI
            physics_timestep: Physics simulation timestep
        """
        # Load config
        if config is None:
            from ..core.config import DEFAULT_SIM_CONFIG
            config = Config.from_dict(DEFAULT_SIM_CONFIG)
        self.config = config

        # Initialize PyBullet
        if gui:
            self.client_id = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Disable GUI panels
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        else:
            self.client_id = p.connect(p.DIRECT)

        # Set search path for PyBullet data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Physics setup
        self.physics_timestep = physics_timestep
        p.setTimeStep(self.physics_timestep, physicsClientId=self.client_id)
        p.setGravity(0, 0, config.get('physics.gravity', -9.81), physicsClientId=self.client_id)

        # Initialize environment
        if environment is None:
            environment = ForestEnvironment(
                client_id=self.client_id,
                tree_spacing=config.get('environment.tree_spacing', 5.0),
                area_size=tuple(config.get('environment.area_size', [100, 100])),
                num_trees=config.get('environment.num_trees', 100),
                seed=config.get('experiment.seed', None)
            )
            environment.generate()

        self.environment = environment

        # Initialize quadrotor
        self.quadrotor = Quadrotor(
            client_id=self.client_id,
            mass=config.get('quadrotor.mass', 0.775),
            arm_length=config.get('quadrotor.arm_length', 0.17),
            max_thrust=config.get('quadrotor.max_thrust', 15.0),
            max_torque=config.get('quadrotor.max_torque', 0.5)
        )

        # Initialize sensors
        self.camera = DepthCamera(
            client_id=self.client_id,
            width=config.get('camera.width', 640),
            height=config.get('camera.height', 480),
            fov=config.get('camera.fov', 90.0),
            near=config.get('camera.near', 0.1),
            far=config.get('camera.far', 100.0)
        )

        self.imu = IMU(
            accel_noise_std=config.get('imu.accel_noise', 0.0),
            gyro_noise_std=config.get('imu.gyro_noise', 0.0)
        )

        # Initialize controller
        self.controller = PIDController(
            mass=config.get('quadrotor.mass', 0.775),
            kp_pos=config.get('controller.position_p', 1.0),
            kd_pos=config.get('controller.position_d', 0.5),
            kp_att=config.get('controller.attitude_p', 3.0),
            kd_att=config.get('controller.attitude_d', 0.3),
            max_thrust=config.get('quadrotor.max_thrust', 15.0),
            max_torque=config.get('quadrotor.max_torque', 0.5)
        )

        # Simulation state
        self.time = 0.0
        self.step_count = 0
        self.running = False

        # Rendering
        self.gui = gui

    def reset(
        self,
        position: Optional[np.ndarray] = None,
        orientation: Optional[Rotation] = None
    ) -> QuadrotorState:
        """
        Reset simulation to initial state.

        Args:
            position: Initial position (defaults to [0, 0, 1.5])
            orientation: Initial orientation (defaults to identity)

        Returns:
            Initial state
        """
        if position is None:
            position = np.array([0.0, 0.0, 1.5])
        if orientation is None:
            orientation = Rotation.identity()

        self.quadrotor.reset(position, orientation)
        self.controller.reset()

        self.time = 0.0
        self.step_count = 0
        self.running = True

        return self.get_state()

    def step(
        self,
        action: Optional[Tuple[float, np.ndarray]] = None,
        target_position: Optional[np.ndarray] = None
    ) -> QuadrotorState:
        """
        Step simulation forward.

        Args:
            action: Optional (thrust, torque) tuple for direct control
            target_position: Optional target position for automatic control

        Returns:
            New state after step
        """
        # Get current state
        state = self.get_state()

        # Compute control action
        if action is None:
            if target_position is not None:
                # Use controller
                thrust, torque = self.controller.compute_control(
                    state=state,
                    target_position=target_position,
                    dt=self.physics_timestep
                )
            else:
                # Hover in place
                thrust, torque = self.controller.compute_control(
                    state=state,
                    target_position=state.position,
                    dt=self.physics_timestep
                )
        else:
            thrust, torque = action

        # Apply action to quadrotor
        self.quadrotor.apply_action(thrust, torque)

        # Step physics
        p.stepSimulation(physicsClientId=self.client_id)

        # Update time
        self.time += self.physics_timestep
        self.step_count += 1

        return self.get_state()

    def get_state(self) -> QuadrotorState:
        """Get current quadrotor state."""
        state = self.quadrotor.get_state()
        state.timestamp = self.time
        return state

    def get_sensor_data(self) -> Dict[str, Any]:
        """
        Get all sensor readings.

        Returns:
            Dictionary with keys: 'rgb', 'depth', 'imu_accel', 'imu_gyro'
        """
        state = self.get_state()

        # Camera
        rgb, depth = self.camera.capture_rgbd(state)

        # IMU
        accel, gyro = self.imu.measure(state)

        return {
            'rgb': rgb,
            'depth': depth,
            'imu_accel': accel,
            'imu_gyro': gyro
        }

    def render(self, camera_distance: float = 5.0) -> None:
        """
        Render the simulation (only works if GUI is enabled).

        Args:
            camera_distance: Distance from quadrotor for following camera
        """
        if not self.gui:
            return

        # Follow quadrotor with camera
        state = self.get_state()
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=state.position,
            physicsClientId=self.client_id
        )

    def run_trajectory(
        self,
        trajectory: Trajectory,
        max_time: float = 30.0,
        realtime: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a trajectory.

        Args:
            trajectory: Trajectory to execute
            max_time: Maximum execution time (seconds)
            realtime: Whether to run in realtime

        Returns:
            Dictionary with execution info (states, collision, success, etc.)
        """
        states = []
        collisions = []
        time_start = time.time()

        traj_time = 0.0

        while traj_time < max_time and self.running:
            # Get control from trajectory
            thrust, torque = self.controller.compute_trajectory_control(
                state=self.get_state(),
                trajectory=trajectory,
                time_in_trajectory=traj_time,
                dt=self.physics_timestep
            )

            # Step simulation
            state = self.step(action=(thrust, torque))
            states.append(state)

            # Check collision
            collision = self.quadrotor.check_collision()
            collisions.append(collision)

            if collision:
                break

            # Render
            if self.gui:
                self.render()

            # Realtime sync
            if realtime:
                elapsed = time.time() - time_start
                if elapsed < self.time:
                    time.sleep(self.time - elapsed)

            traj_time += self.physics_timestep

        return {
            'states': states,
            'collisions': collisions,
            'success': not any(collisions),
            'trajectory_time': traj_time,
            'num_steps': len(states)
        }

    def check_collision(self) -> bool:
        """Check if quadrotor is in collision."""
        return self.quadrotor.check_collision()

    def get_point_cloud(self, radius: float = 20.0) -> np.ndarray:
        """
        Get point cloud of environment around quadrotor.

        Args:
            radius: Radius around quadrotor to sample

        Returns:
            Point cloud array (N, 3)
        """
        state = self.get_state()
        min_bound = state.position - radius
        max_bound = state.position + radius

        return self.environment.get_point_cloud(bounds=(min_bound, max_bound))

    def close(self) -> None:
        """Cleanup and close simulator."""
        self.running = False
        p.disconnect(physicsClientId=self.client_id)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass
