#!/usr/bin/env python
"""
Demo script showing basic simulation usage.

This demonstrates:
- Creating a simulator with forest environment
- Resetting to initial position
- Hovering in place
- Moving to target positions
- Reading sensor data
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from agile_autonomy.simulation import QuadrotorSimulator, ForestEnvironment
from agile_autonomy.core import Config, Trajectory
import time


def demo_hover():
    """Demo 1: Simple hovering."""
    print("=" * 60)
    print("DEMO 1: Hovering in Place")
    print("=" * 60)

    # Load config
    config = Config.from_yaml("config/sim_config.yaml")

    # Create simulator with GUI
    with QuadrotorSimulator(config=config, gui=True) as sim:
        # Reset to starting position
        state = sim.reset(position=np.array([0, 0, 2.0]))
        print(f"Initial state:\n{state}\n")

        # Hover for 5 seconds
        print("Hovering for 5 seconds...")
        num_steps = int(5.0 / sim.physics_timestep)

        for i in range(num_steps):
            # Step with no target (will hover in place)
            state = sim.step()

            # Render every 10 steps
            if i % 10 == 0:
                sim.render()

            # Print state every second
            if i % int(1.0 / sim.physics_timestep) == 0:
                print(f"t={sim.time:.2f}s: pos={state.position}, alt={state.altitude:.2f}m")

            # Small delay for realtime visualization
            time.sleep(sim.physics_timestep)

        print(f"\nFinal state:\n{state}")


def demo_waypoint_navigation():
    """Demo 2: Navigate to waypoints."""
    print("\n" + "=" * 60)
    print("DEMO 2: Waypoint Navigation")
    print("=" * 60)

    config = Config.from_yaml("config/sim_config.yaml")

    with QuadrotorSimulator(config=config, gui=True) as sim:
        # Reset
        sim.reset(position=np.array([0, 0, 2.0]))

        # Define waypoints
        waypoints = [
            np.array([5.0, 0.0, 2.0]),
            np.array([5.0, 5.0, 2.5]),
            np.array([0.0, 5.0, 2.0]),
            np.array([0.0, 0.0, 2.0]),
        ]

        print(f"Flying to {len(waypoints)} waypoints...")

        for i, waypoint in enumerate(waypoints):
            print(f"\nWaypoint {i+1}/{len(waypoints)}: {waypoint}")

            # Fly to waypoint (stop when close)
            while True:
                state = sim.step(target_position=waypoint)

                # Check if reached
                distance = np.linalg.norm(state.position - waypoint)
                if distance < 0.2:  # Within 20cm
                    print(f"  Reached! (distance: {distance:.3f}m)")
                    break

                # Check timeout
                if sim.time > 60.0:
                    print("  Timeout!")
                    break

                # Render
                sim.render()
                time.sleep(sim.physics_timestep)

        print("\nWaypoint navigation complete!")


def demo_trajectory_following():
    """Demo 3: Follow a predefined trajectory."""
    print("\n" + "=" * 60)
    print("DEMO 3: Trajectory Following")
    print("=" * 60)

    config = Config.from_yaml("config/sim_config.yaml")

    with QuadrotorSimulator(config=config, gui=True) as sim:
        # Reset
        sim.reset(position=np.array([0, 0, 2.0]))

        # Create a circular trajectory
        t = np.linspace(0, 2*np.pi, 50)
        radius = 3.0
        height = 2.0

        positions = np.column_stack([
            radius * np.cos(t),
            radius * np.sin(t),
            height * np.ones_like(t)
        ])

        trajectory = Trajectory.from_positions(positions, dt=0.1, frame='world')

        print(f"Following circular trajectory ({len(trajectory)} points)...")

        # Execute trajectory
        result = sim.run_trajectory(trajectory, max_time=20.0, realtime=True)

        print(f"\nTrajectory execution:")
        print(f"  Success: {result['success']}")
        print(f"  Time: {result['trajectory_time']:.2f}s")
        print(f"  Steps: {result['num_steps']}")
        print(f"  Collisions: {sum(result['collisions'])}")


def demo_sensors():
    """Demo 4: Read sensor data."""
    print("\n" + "=" * 60)
    print("DEMO 4: Sensor Data")
    print("=" * 60)

    config = Config.from_yaml("config/sim_config.yaml")

    with QuadrotorSimulator(config=config, gui=True) as sim:
        # Reset
        state = sim.reset(position=np.array([0, 0, 2.0]))

        print("Reading sensor data...")

        # Hover and read sensors
        for i in range(10):
            sim.step()

            if i % 2 == 0:  # Every other step
                sensors = sim.get_sensor_data()

                print(f"\nTime: {sim.time:.2f}s")
                print(f"  RGB image: {sensors['rgb'].shape}")
                print(f"  Depth image: {sensors['depth'].shape}")
                print(f"  Depth range: [{sensors['depth'].min():.2f}, {sensors['depth'].max():.2f}]m")
                print(f"  IMU accel: {sensors['imu_accel']}")
                print(f"  IMU gyro: {sensors['imu_gyro']}")

                # Save first image
                if i == 0:
                    import cv2
                    cv2.imwrite("demo_rgb.png", sensors['rgb'][:, :, ::-1])  # BGR for OpenCV
                    print("  Saved RGB image to demo_rgb.png")

            sim.render()
            time.sleep(sim.physics_timestep * 10)


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("AGILE AUTONOMY V2 - SIMULATION DEMOS")
    print("=" * 60 + "\n")

    demos = [
        ("Hover", demo_hover),
        ("Waypoint Navigation", demo_waypoint_navigation),
        ("Trajectory Following", demo_trajectory_following),
        ("Sensors", demo_sensors),
    ]

    print("Available demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print("  5. All demos")
    print("  0. Exit")

    choice = input("\nSelect demo (0-5): ").strip()

    if choice == "0":
        print("Exiting.")
        return
    elif choice == "5":
        for name, demo_func in demos:
            demo_func()
    elif choice in ["1", "2", "3", "4"]:
        idx = int(choice) - 1
        demos[idx][1]()
    else:
        print("Invalid choice!")
        return

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
