import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PPO1(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc = nn.Linear(num_inputs, 128)
        self.fc_actor = nn.Linear(128, num_outputs)
        self.fc_critic = nn.Linear(128, 1)
        self.model_name = "PPO_clip"

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")

    def forward(self, inputs):
        x = F.relu(self.fc(inputs))
        policy = F.softmax(self.fc_actor(x), dim=-1)
        value = self.fc_critic(x)
        return policy, value

    def get_gae(self, values, next_values, rewards, masks, gamma, lambda_gae):
        advantages = torch.zeros_like(rewards)
        running_advantage = 0.0

        for index in range(len(rewards) - 1, -1, -1):
            delta = rewards[index] + gamma * next_values[index] * masks[index] - values[index]
            running_advantage = delta + gamma * lambda_gae * running_advantage * masks[index]
            advantages[index] = running_advantage

        returns = advantages + values
        return returns, advantages

    @classmethod
    def train_model(
        cls,
        net,
        optimizer,
        batch,
        gamma,
        lambda_gae,
        critic_coefficient,
        entropy_coefficient,
        epoch=10,
        batch_size=64,
        clips_eps=0.1,
    ):
        device = next(net.parameters()).device

        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        actions = torch.stack(batch.action).to(device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).view(-1)
        masks = torch.tensor(batch.mask, dtype=torch.float32, device=device).view(-1)

        old_policies, old_values = net(states)
        _, old_next_values = net(next_states)

        old_policies = old_policies.view(-1, net.num_outputs)
        old_values = old_values.view(-1)
        old_next_values = old_next_values.view(-1)
        old_log_prob = (torch.log(old_policies + 1e-8) * actions).sum(dim=1).detach()

        returns, advantages = net.get_gae(
            old_values.detach(),
            old_next_values.detach(),
            rewards,
            masks,
            gamma,
            lambda_gae,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        dataset_size = states.size(0)
        for _ in range(epoch):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, batch_size):
                batch_idx = indices[start : start + batch_size]
                sample_states = states[batch_idx]
                sample_actions = actions[batch_idx]
                sample_advantages = advantages[batch_idx]
                sample_returns = returns[batch_idx]
                sample_old_log = old_log_prob[batch_idx]

                policies, values = net(sample_states)
                policies = policies.view(-1, net.num_outputs)
                log_policies = (torch.log(policies + 1e-8) * sample_actions).sum(dim=1)
                ratio = torch.exp(log_policies - sample_old_log)

                surr1 = ratio * sample_advantages
                surr2 = torch.clamp(ratio, 1 - clips_eps, 1 + clips_eps) * sample_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values.view(-1), sample_returns)
                entropy = -(policies * torch.log(policies + 1e-8)).sum(dim=1).mean()

                loss = actor_loss + critic_coefficient * critic_loss - entropy_coefficient * entropy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return loss

    def get_action(self, inputs):
        with torch.no_grad():
            policy, _ = self.forward(inputs)
            policy = policy.squeeze(0).detach().cpu().numpy()
            return np.random.choice(self.num_outputs, 1, p=policy)[0]


class PPO2(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc = nn.Linear(num_inputs, 128)
        self.fc_actor = nn.Linear(128, num_outputs)
        self.fc_critic = nn.Linear(128, 1)
        self.model_name = "PPO2_penalty"

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")

    def forward(self, inputs):
        x = F.relu(self.fc(inputs))
        policy = F.softmax(self.fc_actor(x), dim=-1)
        value = self.fc_critic(x)
        return policy, value

    def get_gae(self, values, next_values, rewards, masks, gamma, lambda_gae):
        advantages = torch.zeros_like(rewards)
        running_advantage = 0.0

        for index in range(len(rewards) - 1, -1, -1):
            delta = rewards[index] + gamma * next_values[index] * masks[index] - values[index]
            running_advantage = delta + gamma * lambda_gae * running_advantage * masks[index]
            advantages[index] = running_advantage

        returns = advantages + values
        return returns, advantages

    @staticmethod
    def kl_divergence(policy, old_policy):
        kl = old_policy * torch.log((old_policy + 1e-8) / (policy + 1e-8))
        return kl.sum(1)

    @classmethod
    def train_model(
        cls,
        net,
        optimizer,
        batch,
        gamma,
        lambda_gae,
        critic_coefficient,
        entropy_coefficient,
        epoch=10,
        batch_size=64,
        beta=0.01,
    ):
        device = next(net.parameters()).device

        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        actions = torch.stack(batch.action).to(device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).view(-1)
        masks = torch.tensor(batch.mask, dtype=torch.float32, device=device).view(-1)

        old_policies, old_values = net(states)
        _, old_next_values = net(next_states)

        old_policies = old_policies.view(-1, net.num_outputs).detach()
        old_values = old_values.view(-1)
        old_next_values = old_next_values.view(-1)
        old_log_prob = (torch.log(old_policies + 1e-8) * actions).sum(dim=1).detach()

        returns, advantages = net.get_gae(
            old_values.detach(),
            old_next_values.detach(),
            rewards,
            masks,
            gamma,
            lambda_gae,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        dataset_size = states.size(0)
        for _ in range(epoch):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, batch_size):
                batch_idx = indices[start : start + batch_size]
                sample_states = states[batch_idx]
                sample_actions = actions[batch_idx]
                sample_advantages = advantages[batch_idx]
                sample_returns = returns[batch_idx]
                sample_old_log = old_log_prob[batch_idx]
                sample_old_policies = old_policies[batch_idx]

                policies, values = net(sample_states)
                policies = policies.view(-1, net.num_outputs)
                log_policies = (torch.log(policies + 1e-8) * sample_actions).sum(dim=1)
                ratio = torch.exp(log_policies - sample_old_log)
                kl = cls.kl_divergence(policies, sample_old_policies)

                actor_loss = -(ratio * sample_advantages - beta * kl).mean()
                critic_loss = F.mse_loss(values.view(-1), sample_returns)
                entropy = -(policies * torch.log(policies + 1e-8)).sum(dim=1).mean()

                loss = actor_loss + critic_coefficient * critic_loss - entropy_coefficient * entropy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return loss

    def get_action(self, inputs):
        with torch.no_grad():
            policy, _ = self.forward(inputs)
            policy = policy.squeeze(0).detach().cpu().numpy()
            return np.random.choice(self.num_outputs, 1, p=policy)[0]
