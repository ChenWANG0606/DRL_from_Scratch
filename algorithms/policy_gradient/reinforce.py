import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Reinforce(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model_name = "Reinforce"

        self.fc_1 = nn.Linear(num_inputs, 128)
        self.fc_2 = nn.Linear(128, num_outputs)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")

    def forward(self, inputs):
        x = F.relu(self.fc_1(inputs))
        policy = F.softmax(self.fc_2(x), dim=-1)
        return policy

    @classmethod
    def train_model(cls, net, transitions, optimizer, gamma):
        device = next(net.parameters()).device
        states, actions, rewards, masks = (
            transitions.state,
            transitions.action,
            transitions.reward,
            transitions.mask,
        )

        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        masks = torch.tensor(masks, dtype=torch.float32, device=device)

        returns = torch.zeros_like(rewards)
        running_return = 0
        for index in reversed(range(len(rewards))):
            running_return = rewards[index] + gamma * running_return * masks[index]
            returns[index] = running_return

        policies = net(states).view(-1, net.num_outputs)
        log_policies = (torch.log(policies) * actions.detach()).sum(dim=1)
        loss = (-log_policies * returns).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def get_action(self, inputs):
        policy = self.forward(inputs)
        policy = policy.squeeze(0).detach().cpu().numpy()
        return np.random.choice(self.num_outputs, 1, p=policy)[0]


class Reinforce2(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model_name = "Reinforce-baseline"

        self.fc_1 = nn.Linear(num_inputs, 128)
        self.fc_2 = nn.Linear(128, num_outputs)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")

    def forward(self, inputs):
        x = F.relu(self.fc_1(inputs))
        policy = F.softmax(self.fc_2(x), dim=-1)
        return policy

    @staticmethod
    def get_ae(rewards, masks, gamma):
        returns = torch.zeros_like(rewards)
        running_return = 0
        for index in reversed(range(len(rewards))):
            running_return = rewards[index] + gamma * running_return * masks[index]
            returns[index] = running_return

        baseline = returns.mean()
        advantages = returns - baseline
        return returns, advantages

    @classmethod
    def train_model(cls, net, transitions, optimizer, gamma):
        device = next(net.parameters()).device
        states, actions, rewards, masks = (
            transitions.state,
            transitions.action,
            transitions.reward,
            transitions.mask,
        )

        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        masks = torch.tensor(masks, dtype=torch.float32, device=device)

        _, advantages = cls.get_ae(rewards, masks, gamma)

        policies = net(states).view(-1, net.num_outputs)
        log_policies = (torch.log(policies) * actions.detach()).sum(dim=1)
        loss = (-log_policies * advantages).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def get_action(self, inputs):
        policy = self.forward(inputs)
        policy = policy.squeeze(0).detach().cpu().numpy()
        return np.random.choice(self.num_outputs, 1, p=policy)[0]
