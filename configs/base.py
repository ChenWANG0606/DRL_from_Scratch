from dataclasses import dataclass

import torch


@dataclass
class BaseConfig:
    gamma: float = 0.99
    lr: float = 0.001
    log_interval: int = 10
    device: torch.device = (
        torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    )
