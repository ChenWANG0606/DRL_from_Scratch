import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GAE(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc = nn.Linear(num_inputs, 128)
        self.fc_actor = nn.Linear(128, num_outputs)
        self.fc_critic = nn.Linear(128, 1)
        self.model_name = "GAE"

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
    def train_model(cls, net, optimizer, batch, gamma, lambda_gae, critic_coefficient, entropy_coefficient):
        device = next(net.parameters()).device

        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        actions = torch.stack(batch.action).to(device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).view(-1)
        masks = torch.tensor(batch.mask, dtype=torch.float32, device=device).view(-1)

        policies, values = net(states)
        _, next_values = net(next_states)

        policies = policies.view(-1, net.num_outputs)
        values = values.view(-1)
        next_values = next_values.view(-1)

        returns, advantages = net.get_gae(
            values.detach(),
            next_values.detach(),
            rewards,
            masks,
            gamma,
            lambda_gae,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        log_policies = (torch.log(policies + 1e-8) * actions.detach()).sum(dim=1)
        actor_loss = -(log_policies * advantages.detach()).mean()
        critic_loss = F.mse_loss(values, returns.detach())
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
