#!/usr/bin/env python
"""
Test/evaluation script for trained PlaNet model.

Usage:
    python scripts/test.py --config config/test_config.yaml
    python scripts/test.py --config config/test_config.yaml --checkpoint data/checkpoints/best_model.pt
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import numpy as np
from agile_autonomy.core import Config, Trajectory
from agile_autonomy.models import PlaNet
from agile_autonomy.simulation import QuadrotorSimulator, ForestEnvironment
from agile_autonomy.utils import set_seed, get_device
from typing import Dict, Any, List
import json
from tqdm import tqdm


class ModelEvaluator:
    """Evaluate trained PlaNet model in simulation."""

    def __init__(self, model: PlaNet, config: Dict[str, Any], device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        self.model.eval()

        # Inference parameters
        self.network_freq = config.get('inference', {}).get('network_frequency', 15.0)
        self.dt_network = 1.0 / self.network_freq

    @torch.no_grad()
    def predict_trajectory(
        self,
        depth_image: np.ndarray,
        state_vector: np.ndarray
    ) -> Trajectory:
        """
        Predict trajectory from current observations.

        Args:
            depth_image: (H, W) or (H, W, 3) depth image
            state_vector: State vector

        Returns:
            Best trajectory
        """
        # Prepare depth
        depth_tensor = torch.tensor(depth_image, dtype=torch.float32)
        if depth_tensor.ndim == 2:
            depth_tensor = depth_tensor.unsqueeze(0).expand(3, -1, -1)  # (3, H, W)
        elif depth_tensor.shape[-1] == 3:
            depth_tensor = depth_tensor.permute(2, 0, 1)  # (3, H, W)

        # Add batch and sequence dimensions
        depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, H, W)
        depth_tensor = depth_tensor.to(self.device)

        # Prepare state
        state_tensor = torch.tensor(state_vector, dtype=torch.float32)
        state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, state_dim)
        state_tensor = state_tensor.to(self.device)

        # Forward pass
        predictions = self.model(state_tensor, depth_tensor)  # (1, modes, output_dim)

        # Get trajectories and alphas
        trajectories, alphas = self.model.predict_trajectories(state_tensor, depth_tensor)
        # trajectories: (1, modes, out_seq_len, state_dim)
        # alphas: (1, modes)

        # Select best mode (lowest alpha = best)
        best_mode_idx = torch.argmin(alphas[0]).item()
        best_trajectory = trajectories[0, best_mode_idx].cpu().numpy()  # (out_seq_len, state_dim)

        # Create Trajectory object
        trajectory = Trajectory.from_positions(
            best_trajectory,
            dt=self.config.get('inference', {}).get('time_step', 0.1),
            frame='body'
        )

        return trajectory

    def evaluate_rollout(
        self,
        simulator: QuadrotorSimulator,
        start_position: np.ndarray,
        max_time: float = 30.0
    ) -> Dict[str, Any]:
        """
        Evaluate model on a single rollout.

        Args:
            simulator: Simulator instance
            start_position: [x, y, z, yaw]
            max_time: Maximum rollout time

        Returns:
            Dictionary with results
        """
        from scipy.spatial.transform import Rotation

        # Reset simulator
        yaw = start_position[3] if len(start_position) > 3 else 0
        orientation = Rotation.from_euler('z', yaw)
        state = simulator.reset(
            position=start_position[:3],
            orientation=orientation
        )

        # Track results
        states = [state]
        collisions = []
        predicted_trajectories = []

        time_elapsed = 0.0
        time_since_prediction = 0.0

        current_trajectory = None
        trajectory_start_time = 0.0

        while time_elapsed < max_time:
            # Check if need new prediction
            if time_since_prediction >= self.dt_network or current_trajectory is None:
                # Get sensor data
                sensors = simulator.get_sensor_data()
                depth_image = sensors['depth']

                # Prepare state vector
                state_vector = state.to_network_input(
                    include_position=self.config['model']['inputs']['position'],
                    include_attitude=self.config['model']['inputs']['attitude'],
                    include_velocity=True,
                    include_bodyrates=self.config['model']['inputs']['bodyrates'],
                    velocity_frame=self.config['model']['inputs']['velocity_frame']
                )

                # Predict trajectory
                current_trajectory = self.predict_trajectory(depth_image, state_vector)
                predicted_trajectories.append(current_trajectory)
                trajectory_start_time = 0.0
                time_since_prediction = 0.0

            # Execute trajectory
            if current_trajectory is not None:
                # Get target from trajectory
                target_point = current_trajectory.get_point_at_time(trajectory_start_time)
                if target_point is not None:
                    # Transform to world frame
                    target_world = state.position + state.rotation_matrix.T @ target_point.position
                else:
                    target_world = state.position

                # Step simulation
                state = simulator.step(target_position=target_world)
            else:
                # Hover
                state = simulator.step()

            # Update time
            simulator.render()
            time_elapsed += simulator.physics_timestep
            time_since_prediction += simulator.physics_timestep
            trajectory_start_time += simulator.physics_timestep

            # Check collision
            collision = simulator.check_collision()
            collisions.append(collision)

            if collision:
                print("  Collision detected!")
                break

            states.append(state)

        # Compute metrics
        success = not any(collisions)
        total_distance = sum(
            np.linalg.norm(states[i+1].position - states[i].position)
            for i in range(len(states) - 1)
        )

        return {
            'success': success,
            'collision': any(collisions),
            'num_collisions': sum(collisions),
            'time': time_elapsed,
            'distance': total_distance,
            'num_states': len(states),
            'num_predictions': len(predicted_trajectories)
        }


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test PlaNet model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/test_config.yaml",
        help="Path to test config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (overrides config)"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show simulation GUI"
    )
    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    config = Config.from_yaml(args.config)
    config_dict = config.to_dict()

    # Set seed
    seed = config_dict.get('experiment', {}).get('seed', 42)
    set_seed(seed)

    # Get device
    device = get_device(use_cuda=config_dict['experiment']['device'] == 'cuda')

    print("\n" + "="*60)
    print("AGILE AUTONOMY - TESTING")
    print("="*60)
    print(f"Experiment: {config_dict['experiment']['name']}")
    print(f"Device: {device}")
    print("="*60 + "\n")

    # Load model
    print("Loading model...")
    model_config = config_dict['model']
    model = PlaNet(model_config)

    # Load checkpoint
    checkpoint_path = args.checkpoint or config_dict['checkpoint']['path']
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

    # Create evaluator
    evaluator = ModelEvaluator(model, config_dict, device)

    # Run tests
    test_config = config_dict.get('test', {})
    num_rollouts = test_config.get('num_rollouts', 10)
    max_time = test_config.get('max_time', 30.0)
    start_positions = test_config.get('start_positions', [[0, 0, 2.0, 0]])

    print(f"\nRunning {num_rollouts} test rollouts...\n")

    all_results = []

    for rollout_idx in tqdm(range(num_rollouts), desc="Testing"):
        # Create simulator
        with QuadrotorSimulator(gui=args.gui) as sim:
            # Random start position
            start_pos = start_positions[rollout_idx % len(start_positions)]
            start_pos = np.array(start_pos)

            # Run evaluation
            result = evaluator.evaluate_rollout(sim, start_pos, max_time)
            result['rollout_idx'] = rollout_idx
            result['start_position'] = start_pos.tolist()

            all_results.append(result)

            print(f"  Rollout {rollout_idx}: Success={result['success']}, "
                  f"Time={result['time']:.1f}s, Distance={result['distance']:.1f}m")

    # Compute summary statistics
    success_rate = sum(r['success'] for r in all_results) / len(all_results)
    avg_time = np.mean([r['time'] for r in all_results])
    avg_distance = np.mean([r['distance'] for r in all_results])
    total_collisions = sum(r['num_collisions'] for r in all_results)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Success Rate: {success_rate*100:.1f}% ({sum(r['success'] for r in all_results)}/{len(all_results)})")
    print(f"Average Time: {avg_time:.2f}s")
    print(f"Average Distance: {avg_distance:.2f}m")
    print(f"Total Collisions: {total_collisions}")
    print("="*60)

    # Save results
    output_dir = Path(config_dict.get('output', {}).get('results_dir', 'results'))
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"{config_dict['experiment']['name']}_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'config': config_dict,
            'summary': {
                'success_rate': success_rate,
                'avg_time': avg_time,
                'avg_distance': avg_distance,
                'total_collisions': total_collisions
            },
            'rollouts': all_results
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
