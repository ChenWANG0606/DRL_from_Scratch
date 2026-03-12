import argparse
from dataclasses import fields, is_dataclass
from typing import Type

import torch


def build_default_configs(config_cls: Type):
    if not is_dataclass(config_cls):
        raise TypeError("config_cls must be a dataclass")

    parser = argparse.ArgumentParser()
    for field in fields(config_cls):
        arg_name = f"--{field.name}"
        default = field.default
        arg_type = field.type
        if arg_type is torch.device:
            parser.add_argument(arg_name, default=default)
        else:
            parser.add_argument(arg_name, type=arg_type, default=default)

    args = parser.parse_args()
    return config_cls(**vars(args))
