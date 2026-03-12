from dataclasses import dataclass

from .base import BaseConfig


@dataclass
class DQNConfig(BaseConfig):
    env_name: str = "CartPole-v1"
    goal_score: int = 200
    sequence_length: int = 4
    replay_memory_capacity: int = 1000
    update_target: int = 100
    lr: float = 0.0001
    initial_exploration: int = 1000
    batch_size: int = 64
    epsilon: float = 0.9
    seed: int = 42
    n_step: int = 1
    alpha: float = 0.6
    beta: float = 0.4
    small_epsilon: float = 0.01


@dataclass
class RainbowConfig(DQNConfig):
    lr: float = 0.0001
    n_step: int = 3
    alpha: float = 0.6
    beta: float = 0.4
    small_epsilon: float = 0.01
    atoms: int = 51
    vmin: float = -10.0
    vmax: float = 10.0
