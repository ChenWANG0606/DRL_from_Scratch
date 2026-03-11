from dataclasses import dataclass
import torch
import argparse
from dataclasses import fields, is_dataclass
from typing import Type


def build_default_configs(config_cls: Type):
    """
    从 dataclass 构建 argparse，并返回 dataclass 实例
    """
    if not is_dataclass(config_cls):
        raise TypeError("config_cls must be a dataclass")

    parser = argparse.ArgumentParser()

    for f in fields(config_cls):
        arg_name = f"--{f.name}"
        default = f.default
        arg_type = f.type

        # torch.device 不能直接 argparse 解析，特殊处理
        if arg_type is torch.device:
            parser.add_argument(arg_name, default=default)
        else:
            parser.add_argument(
                arg_name,
                type=arg_type,
                default=default,
            )

    args = parser.parse_args()

    # Namespace → dataclass
    config = config_cls(**vars(args))
    return config


@dataclass
class BaseConfig:
    gamma: float = 0.99
    lr: float = 0.001
    log_interval: int = 10
    device: torch.device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )


@dataclass
class RFConfig(BaseConfig):
    env_name: str = "CartPole-v1"
    goal_score: int = 200


@dataclass
class ACConfig(BaseConfig):
    env_name: str = "CartPole-v1"
    goal_score: int = 200

@dataclass
class TNPGConfig(BaseConfig):
    env_name: str = "CartPole-v1"
    goal_score: int = 200
    lr: float = 1.0
    max_kl: float = 0.01
    cg_damp: float = 0.1
    cg_iters: int = 10

@dataclass
class GAEConfig(BaseConfig):
    env_name: str = "CartPole-v1"
    goal_score: int = 200
    gamma: float = 0.99
    critic_coefficient: float = 1.0
    entropy_coefficient: float = 0.01
    lambda_gae: float = 0.95

@dataclass
class DQNConfig(BaseConfig):
    env_name: str = "CartPole-v1"
    goal_score: int = 200
    sequence_length: int = 4        # 将多个历史时刻拼接起来
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
