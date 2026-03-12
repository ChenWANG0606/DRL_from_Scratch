from dataclasses import dataclass

from .base import BaseConfig


@dataclass
class ACConfig(BaseConfig):
    env_name: str = "CartPole-v1"
    goal_score: int = 200


@dataclass
class GAEConfig(BaseConfig):
    env_name: str = "CartPole-v1"
    goal_score: int = 200
    gamma: float = 0.99
    critic_coefficient: float = 1.0
    entropy_coefficient: float = 0.01
    lambda_gae: float = 0.95
