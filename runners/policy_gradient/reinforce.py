import torch
import torch.optim as optim

from algorithms.common import Memory
from algorithms.policy_gradient import Reinforce2
from configs import RFConfig, build_default_configs
from runners.common import build_writer, make_env, set_seeds, update_running_score
from utils.train_utils import save_train_plot

def main(args):
    env_name = args.env_name
    seed = getattr(args, "seed", 42)
    env = make_env(env_name, use_gymnasium=True)
    set_seeds(env, seed=seed)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n

    print('state size:', num_inputs)
    print('action size:', num_actions)

    net = Reinforce2(num_inputs, num_actions)
    model_name = net.model_name

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    writer = build_writer(model_name, env_name, seed)

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
        memory = Memory()# 每一轮重置Memory，因为策略梯度参数更新后，之前的轨迹就过时了

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

        loss = Reinforce2.train_model(net, memory.sample(), optimizer, args.gamma)
        loss_history.append(float(loss.item()))
            

        score = score if score == 500.0 else score + 1
        # score是用来给每一次游戏打分的，超过goal_score
        # 游戏就赢了
        running_score = update_running_score(running_score, score)
        reward_history.append(float(running_score))
        if e % args.log_interval == 0:
            print("{} episode | score: {:.2f}".format(e, running_score))
            writer.add_scalar("log/score", float(running_score), e)
            writer.add_scalar("log/loss", float(loss), e)

        if running_score > args.goal_score:
            break

    save_train_plot(loss_history, reward_history, model_name, seed, env_name)


if __name__ == "__main__":
    args = build_default_configs(RFConfig)
    main(args)
