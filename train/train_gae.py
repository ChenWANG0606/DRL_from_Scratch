
import sys
import os

# Add project root (parent directory of /train) to PYTHONPATH
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import gymnasium as gym
import random
import numpy as np

import torch    
import torch.optim as optim
import torch.nn.functional as F
from algo.model import GAE
from tensorboardX import SummaryWriter


from algo.model_utils import Memory
from configs.configs import GAEConfig, build_default_configs
from utils.train_utils import save_train_plot

def set_seeds(env, seed = 42):
    env.reset(seed=seed)
    env.action_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def main(args):
    env_name = "CartPole-v1"
    seed = 42
    env = gym.make(env_name)
    set_seeds(env, seed=seed)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n

    print('state size:', num_inputs)
    print('action size:', num_actions)

    net = GAE(num_inputs, num_actions)
    model_name = net.model_name

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    writer = SummaryWriter('logs')

    net.to(args.device)
    net.train()
    running_score = 0
    steps = 0
    loss = 0
    reward_history = []
    loss_history = []
    episodes = 3000

    for e in range(episodes):
        done = False
        memory = Memory()

        score = 0
        state, _ = env.reset()
        state = torch.Tensor(state).to(args.device)
        state = state.unsqueeze(0)

        while not done:
            steps += 1

            action = net.get_action(state)
            step_out = env.step(action)
            if len(step_out) == 5:
                next_state, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_out

            next_state = torch.Tensor(next_state).to(args.device)
            next_state = next_state.unsqueeze(0)

            mask = 0 if done else 1
            reward = reward if not done or score == 499 else -1

            action_one_hot = torch.zeros(2)
            action_one_hot[action] = 1
            memory.push(state, next_state, action_one_hot, reward, mask)
    
            score += reward# reward是环境给的只有0-1代表是否还活着s
            state = next_state

        loss = GAE.train_model(net, optimizer, memory.sample(), args.gamma,args.lambda_gae,  args.critic_coefficient, args.entropy_coefficient)
        loss_history.append(float(loss.item()))
            

        score = score if score == 500.0 else score + 1
        # score是用来给每一次游戏打分的，超过goal_score
        # 游戏就赢了
        running_score = 0.99 * running_score + 0.01 * score
        reward_history.append(float(running_score))
        if e % args.log_interval == 0:
            print('{} episode | score: {:.2f}'.format(
                e, running_score))
            writer.add_scalar('log/score', float(running_score), e)
            writer.add_scalar('log/loss', float(loss), e)

        if running_score > args.goal_score:
            break

    save_train_plot(loss_history, reward_history, model_name, seed, env_name)


if __name__=="__main__":
    args = build_default_configs(GAEConfig)
    main(args)
