import gym
import numpy as np
import sys
import os

# Add project root (parent directory of /train) to PYTHONPATH
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.optim as optim
from algo.model import Rainbow
from algo.model_utils import Memory
from tensorboardX import SummaryWriter

from configs.configs import RainbowConfig, build_default_configs
from collections import deque
from utils.train_utils import save_train_plot


def get_action(states_series, sequence_length, policy_net, env):
    if len(states_series) < sequence_length:
        return env.action_space.sample()

    policy_net.reset_noise()
    return policy_net.get_action(torch.stack(list(states_series)))


def sync_target_network(online_net, target_net):
    target_net.load_state_dict(online_net.state_dict())


def train_step(online_net, target_net, optimizer, memory, args):
    batch, indices, weights = memory.priority_sample(args.batch_size, args.beta)
    loss, td_error = Rainbow.train_model(
        online_net,
        target_net,
        optimizer,
        batch,
        args.gamma,
        n_step=args.n_step,
        weights=torch.tensor(weights, dtype=torch.float32, device=args.device),
    )
    memory.update_priority(indices, td_error, args.alpha)
    return loss


def state_to_partial_observability(state):
    state = state[[0, 2]]
    return state


def main(args):
    env = gym.make(args.env_name)
    torch.manual_seed(args.seed)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n

    print("state size:", num_inputs)
    print("action size:", num_actions)

    online_net = Rainbow(
        num_inputs,
        num_actions,
        args.sequence_length,
        atoms=args.atoms,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    target_net = Rainbow(
        num_inputs,
        num_actions,
        args.sequence_length,
        atoms=args.atoms,
        vmin=args.vmin,
        vmax=args.vmax,
    )

    sync_target_network(online_net, target_net)

    optimizer = optim.Adam(online_net.parameters(), lr=args.lr)
    writer = SummaryWriter("logs")

    online_net.to(args.device)
    target_net.to(args.device)
    online_net.train()
    target_net.train()

    memory = Memory(
        args.replay_memory_capacity,
        n_step=args.n_step,
        gamma=args.gamma,
        with_priority=True,
        epsilon=args.small_epsilon,
    )
    running_score = 0
    steps = 0
    loss = 0

    reward_history = []
    loss_history = []

    for e in range(3000):
        done = False
        state_series = deque(maxlen=args.sequence_length)
        next_state_series = deque(maxlen=args.sequence_length)
        score = 0
        state, _ = env.reset()

        # state = state_to_partial_observability(state)
        state = torch.tensor(state, dtype=torch.float32).to(args.device)
        next_state_series.append(state)

        while not done:
            steps += 1
            state_series.append(state)
            action = get_action(state_series, args.sequence_length, online_net, env)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # next_state = state_to_partial_observability(next_state)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(args.device)
            next_state_series.append(next_state)
            done = terminated or truncated
            done_mask = 0 if done else 1
            reward = reward if not done or score == 499 else -1
            action_one_hot = np.zeros(num_actions)
            action_one_hot[action] = 1

            if len(state_series) >= args.sequence_length:
                state_tensor = torch.stack(list(state_series))
                next_state_tensor = torch.stack(list(next_state_series))
                memory.push(state_tensor, next_state_tensor, action_one_hot, reward, done_mask)

            score += reward
            state = next_state

            if steps > args.initial_exploration and len(memory) >= args.batch_size:
                loss = train_step(online_net, target_net, optimizer, memory, args)
                loss_history.append(float(loss.item()))
                if steps % args.update_target == 0:
                    sync_target_network(online_net, target_net)

        score = score if score == 500 else score + 1
        if running_score == 0:
            running_score = score
        else:
            running_score = 0.99 * running_score + 0.01 * score
        reward_history.append(float(running_score))
        if e % args.log_interval == 0:
            print("{} episode | score: {:.2f}".format(e, running_score))
            writer.add_scalar("log/score", float(running_score), e)
            writer.add_scalar("log/loss", float(loss), e)
        if running_score > args.goal_score:
            break

    save_train_plot(loss_history, reward_history, online_net.model_name, args.seed, args.env_name)


if __name__ == "__main__":
    args = build_default_configs(RainbowConfig)
    main(args)
