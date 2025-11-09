"""
Agile Autonomy v2 - Modern Python implementation for high-speed quadrotor navigation.

This package provides a complete implementation of the Agile Autonomy system
without ROS dependencies, using modern Python libraries.
"""

__version__ = "2.0.0"
__author__ = "Agile Autonomy Contributors"

from .core.state import QuadrotorState
from .core.trajectory import Trajectory
from .core.config import Config

__all__ = [
    "QuadrotorState",
    "Trajectory",
    "Config",
]
