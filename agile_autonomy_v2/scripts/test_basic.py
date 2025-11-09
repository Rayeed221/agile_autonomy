#!/usr/bin/env python
"""
Basic tests to verify installation and functionality.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from agile_autonomy.core import QuadrotorState, Trajectory, Config
from agile_autonomy.simulation import QuadrotorSimulator
from scipy.spatial.transform import Rotation


def test_state():
    """Test state representation."""
    print("Testing QuadrotorState...")

    state = QuadrotorState(
        position=np.array([1.0, 2.0, 3.0]),
        velocity=np.array([0.5, 0.0, 0.0]),
        orientation=Rotation.from_euler('xyz', [0, 0, np.pi/4]),
        angular_velocity=np.array([0.1, 0.0, 0.0]),
        timestamp=1.5
    )

    assert state.altitude == 3.0
    assert len(state.quaternion) == 4
    assert state.rotation_matrix.shape == (3, 3)
    assert len(state.euler_angles) == 3

    print("  ✓ QuadrotorState works")


def test_trajectory():
    """Test trajectory representation."""
    print("Testing Trajectory...")

    positions = np.array([
        [0, 0, 1],
        [1, 0, 1],
        [2, 0, 1],
        [3, 0, 1],
    ])

    traj = Trajectory.from_positions(positions, dt=0.1)

    assert len(traj) == 4
    assert traj[0].position[0] == 0
    assert traj[3].position[0] == 3

    # Test interpolation
    point = traj.get_point_at_time(0.15)
    assert point is not None
    assert 1.0 <= point.position[0] <= 2.0

    print("  ✓ Trajectory works")


def test_config():
    """Test configuration."""
    print("Testing Config...")

    config = Config.from_dict({
        'model': {
            'name': 'planet',
            'layers': 3
        },
        'training': {
            'epochs': 100
        }
    })

    assert config.get('model.name') == 'planet'
    assert config.get('model.layers') == 3
    assert config.get('training.epochs') == 100

    print("  ✓ Config works")


def test_simulator_creation():
    """Test simulator initialization."""
    print("Testing Simulator creation...")

    # Create without GUI
    with QuadrotorSimulator(gui=False) as sim:
        state = sim.reset()

        assert state.position[2] > 0  # Should be above ground
        assert sim.time == 0.0

    print("  ✓ Simulator creation works")


def test_simulator_step():
    """Test simulation stepping."""
    print("Testing Simulator stepping...")

    with QuadrotorSimulator(gui=False) as sim:
        initial_state = sim.reset(position=np.array([0, 0, 2.0]))

        # Step a few times
        for _ in range(10):
            state = sim.step()

        assert sim.time > 0
        assert state.timestamp > 0

    print("  ✓ Simulator stepping works")


def test_controller():
    """Test controller."""
    print("Testing Controller...")

    with QuadrotorSimulator(gui=False) as sim:
        sim.reset(position=np.array([0, 0, 2.0]))

        target = np.array([1.0, 0.0, 2.0])

        # Move towards target
        for _ in range(1000):
            state = sim.step(target_position=target)

            # Check if we're getting close
            distance = np.linalg.norm(state.position - target)
            if distance < 0.3:
                break

        # Should reach within 1 meter at least
        final_distance = np.linalg.norm(state.position - target)
        assert final_distance < 1.0, f"Controller didn't converge (dist={final_distance:.2f})"

    print("  ✓ Controller works")


def test_sensors():
    """Test sensors."""
    print("Testing Sensors...")

    with QuadrotorSimulator(gui=False) as sim:
        sim.reset(position=np.array([0, 0, 2.0]))

        sensors = sim.get_sensor_data()

        assert 'rgb' in sensors
        assert 'depth' in sensors
        assert 'imu_accel' in sensors
        assert 'imu_gyro' in sensors

        assert sensors['rgb'].shape[2] == 3  # RGB channels
        assert sensors['depth'].ndim == 2  # 2D depth image
        assert len(sensors['imu_accel']) == 3
        assert len(sensors['imu_gyro']) == 3

    print("  ✓ Sensors work")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RUNNING BASIC TESTS")
    print("=" * 60 + "\n")

    tests = [
        test_state,
        test_trajectory,
        test_config,
        test_simulator_creation,
        test_simulator_step,
        test_controller,
        test_sensors,
    ]

    failed = []

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ✗ {test.__name__} failed: {e}")
            failed.append(test.__name__)

    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED: {len(failed)}/{len(tests)} tests")
        for name in failed:
            print(f"  - {name}")
    else:
        print(f"SUCCESS: All {len(tests)} tests passed!")
    print("=" * 60 + "\n")

    return len(failed) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
