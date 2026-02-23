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

        # torch.device 不能直接 argparse 解析，跳过或特殊处理
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
    goal_score: int = 200


@dataclass
class ACConfig(BaseConfig):
    lr: float = 0.0001
    goal_score: int = 200
