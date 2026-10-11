"""WALL-E and EVE's Signal: an autonomous search agent in a simulated environment."""

__version__ = "1.0.0"

from .env import MissionEnv
from .agent import Agent
from .mission import run_mission

__all__ = ["MissionEnv", "Agent", "run_mission", "__version__"]
