"""Environment generation for simulation."""

import numpy as np
import pybullet as p
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod


class Environment(ABC):
    """Base class for simulation environments."""

    def __init__(self, client_id: int):
        """
        Initialize environment.

        Args:
            client_id: PyBullet physics client ID
        """
        self.client_id = client_id
        self.object_ids = []

    @abstractmethod
    def generate(self) -> None:
        """Generate the environment."""
        pass

    def clear(self) -> None:
        """Remove all environment objects."""
        for obj_id in self.object_ids:
            try:
                p.removeBody(obj_id, physicsClientId=self.client_id)
            except:
                pass
        self.object_ids = []

    def get_point_cloud(self, bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
        """
        Get point cloud representation of the environment.

        Args:
            bounds: Optional tuple of (min_xyz, max_xyz) to limit the region

        Returns:
            Point cloud as array of shape (N, 3)
        """
        points = []

        for obj_id in self.object_ids:
            # Get AABB (Axis-Aligned Bounding Box)
            aabb_min, aabb_max = p.getAABB(obj_id, physicsClientId=self.client_id)

            # Sample points on the surface
            # For simplicity, we'll sample points uniformly in the bounding box
            # In a more sophisticated implementation, you'd use the actual mesh
            num_samples = 100
            for _ in range(num_samples):
                point = np.random.uniform(aabb_min, aabb_max)

                # Filter by bounds if provided
                if bounds is not None:
                    min_bound, max_bound = bounds
                    if np.all(point >= min_bound) and np.all(point <= max_bound):
                        points.append(point)
                else:
                    points.append(point)

        if len(points) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        return np.array(points, dtype=np.float32)

    def __del__(self):
        """Cleanup when destroyed."""
        self.clear()


class ForestEnvironment(Environment):
    """
    Procedurally generated forest environment with trees.

    Trees are represented as vertical cylinders.
    """

    def __init__(
        self,
        client_id: int,
        tree_spacing: float = 5.0,
        tree_spacing_std: float = 1.0,
        area_size: Tuple[float, float] = (100.0, 100.0),
        num_trees: Optional[int] = None,
        tree_radius: float = 0.3,
        tree_height: float = 8.0,
        seed: Optional[int] = None
    ):
        """
        Initialize forest environment.

        Args:
            client_id: PyBullet client ID
            tree_spacing: Average spacing between trees (meters)
            tree_spacing_std: Standard deviation of spacing
            area_size: Size of the area (width, length) in meters
            num_trees: Number of trees (if None, computed from spacing)
            tree_radius: Radius of tree trunks
            tree_height: Height of trees
            seed: Random seed for reproducibility
        """
        super().__init__(client_id)

        self.tree_spacing = tree_spacing
        self.tree_spacing_std = tree_spacing_std
        self.area_size = area_size
        self.tree_radius = tree_radius
        self.tree_height = tree_height
        self.seed = seed

        # Compute number of trees if not provided
        if num_trees is None:
            area = area_size[0] * area_size[1]
            tree_area = tree_spacing ** 2
            num_trees = int(area / tree_area)

        self.num_trees = num_trees

        # Set random seed
        if seed is not None:
            np.random.seed(seed)

    def generate(self) -> None:
        """Generate the forest."""
        self.clear()

        # Create ground plane
        ground_id = p.createCollisionShape(
            p.GEOM_PLANE,
            physicsClientId=self.client_id
        )
        ground_visual = p.createVisualShape(
            p.GEOM_PLANE,
            rgbaColor=[0.3, 0.5, 0.3, 1.0],
            physicsClientId=self.client_id
        )
        ground_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=ground_id,
            baseVisualShapeIndex=ground_visual,
            physicsClientId=self.client_id
        )
        self.object_ids.append(ground_body)

        # Generate tree positions using Poisson disk sampling (simplified)
        tree_positions = self._generate_tree_positions()

        # Create trees
        for pos in tree_positions:
            tree_id = self._create_tree(pos)
            self.object_ids.append(tree_id)

    def _generate_tree_positions(self) -> List[np.ndarray]:
        """Generate tree positions with approximate uniform spacing."""
        positions = []

        # Simple grid-based generation with noise
        grid_spacing = self.tree_spacing
        width, length = self.area_size

        x_positions = np.arange(-width/2, width/2, grid_spacing)
        y_positions = np.arange(-length/2, length/2, grid_spacing)

        for x in x_positions:
            for y in y_positions:
                # Add random offset
                x_offset = np.random.normal(0, self.tree_spacing_std)
                y_offset = np.random.normal(0, self.tree_spacing_std)

                # Randomly skip some trees
                if np.random.rand() > 0.8:  # 20% chance to skip
                    continue

                positions.append(np.array([
                    x + x_offset,
                    y + y_offset,
                    self.tree_height / 2  # z position (center of cylinder)
                ]))

                if len(positions) >= self.num_trees:
                    break
            if len(positions) >= self.num_trees:
                break

        return positions

    def _create_tree(self, position: np.ndarray) -> int:
        """Create a single tree (cylinder)."""
        # Create collision shape
        col_shape = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=self.tree_radius,
            height=self.tree_height,
            physicsClientId=self.client_id
        )

        # Create visual shape (brown trunk)
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=self.tree_radius,
            length=self.tree_height,
            rgbaColor=[0.4, 0.2, 0.1, 1.0],
            physicsClientId=self.client_id
        )

        # Create the tree body (static)
        tree_id = p.createMultiBody(
            baseMass=0,  # Static object
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            physicsClientId=self.client_id
        )

        return tree_id


class ObstacleEnvironment(Environment):
    """
    Environment with various obstacles (boxes, spheres, etc.).
    """

    def __init__(
        self,
        client_id: int,
        obstacle_spacing: float = 5.0,
        area_size: Tuple[float, float] = (100.0, 100.0),
        num_obstacles: Optional[int] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize obstacle environment.

        Args:
            client_id: PyBullet client ID
            obstacle_spacing: Average spacing between obstacles
            area_size: Size of the area (width, length)
            num_obstacles: Number of obstacles
            seed: Random seed
        """
        super().__init__(client_id)

        self.obstacle_spacing = obstacle_spacing
        self.area_size = area_size
        self.seed = seed

        if num_obstacles is None:
            area = area_size[0] * area_size[1]
            obstacle_area = obstacle_spacing ** 2
            num_obstacles = int(area / obstacle_area)

        self.num_obstacles = num_obstacles

        if seed is not None:
            np.random.seed(seed)

    def generate(self) -> None:
        """Generate obstacles."""
        self.clear()

        # Create ground
        ground_id = p.createCollisionShape(
            p.GEOM_PLANE,
            physicsClientId=self.client_id
        )
        ground_visual = p.createVisualShape(
            p.GEOM_PLANE,
            rgbaColor=[0.5, 0.5, 0.5, 1.0],
            physicsClientId=self.client_id
        )
        ground_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=ground_id,
            baseVisualShapeIndex=ground_visual,
            physicsClientId=self.client_id
        )
        self.object_ids.append(ground_body)

        # Generate obstacles
        for _ in range(self.num_obstacles):
            obstacle_id = self._create_random_obstacle()
            self.object_ids.append(obstacle_id)

    def _create_random_obstacle(self) -> int:
        """Create a random obstacle (box or cylinder)."""
        # Random position
        width, length = self.area_size
        x = np.random.uniform(-width/2, width/2)
        y = np.random.uniform(-length/2, length/2)

        # Random type
        obstacle_type = np.random.choice(['box', 'cylinder'])

        if obstacle_type == 'box':
            # Random box size
            half_extents = np.random.uniform([0.3, 0.3, 0.5], [1.0, 1.0, 2.0])
            z = half_extents[2]  # Place on ground

            col_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                physicsClientId=self.client_id
            )
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=[0.6, 0.6, 0.6, 1.0],
                physicsClientId=self.client_id
            )

        else:  # cylinder
            radius = np.random.uniform(0.3, 0.8)
            height = np.random.uniform(1.0, 3.0)
            z = height / 2

            col_shape = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=radius,
                height=height,
                physicsClientId=self.client_id
            )
            visual_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=radius,
                length=height,
                rgbaColor=[0.7, 0.7, 0.7, 1.0],
                physicsClientId=self.client_id
            )

        # Create body
        obstacle_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[x, y, z],
            physicsClientId=self.client_id
        )

        return obstacle_id
