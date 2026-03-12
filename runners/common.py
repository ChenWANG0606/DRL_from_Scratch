import os
import random

import numpy as np
import torch

try:
    from tensorboardX import SummaryWriter
except ModuleNotFoundError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError:
        class SummaryWriter:  # type: ignore[override]
            def __init__(self, *args, **kwargs):
                pass

            def add_scalar(self, *args, **kwargs):
                pass

            def close(self):
                pass

try:
    import gym
except ModuleNotFoundError:
    gym = None

try:
    import gymnasium
except ModuleNotFoundError:
    gymnasium = None


def make_env(env_name, use_gymnasium=False):
    if use_gymnasium and gymnasium is not None:
        return gymnasium.make(env_name)
    if not use_gymnasium and gym is not None:
        return gym.make(env_name)
    if gymnasium is not None:
        return gymnasium.make(env_name)
    if gym is not None:
        return gym.make(env_name)
    raise ModuleNotFoundError("Neither gym nor gymnasium is installed.")


def set_seeds(env, seed=42):
    try:
        env.reset(seed=seed)
    except TypeError:
        env.reset()
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sync_target_network(online_net, target_net):
    target_net.load_state_dict(online_net.state_dict())


def update_running_score(running_score, score):
    if running_score == 0:
        return score
    return 0.99 * running_score + 0.01 * score


def build_writer(model_name, env_name, seed):
    safe_model_name = model_name.replace(" ", "_")
    log_dir = os.path.join("outputs", "tensorboard", f"{env_name}_{safe_model_name}_seed{seed}")
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)
