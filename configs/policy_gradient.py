from dataclasses import dataclass

from .base import BaseConfig


@dataclass
class RFConfig(BaseConfig):
    env_name: str = "CartPole-v1"
    goal_score: int = 200


@dataclass
class TNPGConfig(BaseConfig):
    env_name: str = "CartPole-v1"
    goal_score: int = 200
    lr: float = 0.0005
    max_kl: float = 0.01
    cg_damp: float = 0.1
    cg_iters: int = 10


@dataclass
class PPOConfig(BaseConfig):
    env_name: str = "CartPole-v1"
    goal_score: int = 200
    gamma: float = 0.99
    critic_coefficient: float = 1.0
    entropy_coefficient: float = 0.01
    lambda_gae: float = 0.95
    clip_eps: float = 0.1
    batch_size: int = 64
    epoch: int = 10
