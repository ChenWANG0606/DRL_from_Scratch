import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ACNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc = nn.Linear(num_inputs, 128)
        self.fc_actor = nn.Linear(128, num_outputs)
        self.fc_critic = nn.Linear(128, num_outputs)
        self.model_name = "One_Step_Actor_Critic"

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")

    def forward(self, inputs):
        x = F.relu(self.fc(inputs))
        policy = F.softmax(self.fc_actor(x), dim=-1)
        q_value = self.fc_critic(x)
        return policy, q_value

    @classmethod
    def train_model(cls, net, transition, optimizer, gamma):
        device = next(net.parameters()).device
        state, next_state, action, reward, mask = transition

        state = state.to(device)
        next_state = next_state.to(device)
        reward = torch.tensor(reward, dtype=torch.float32, device=device)
        mask = torch.tensor(mask, dtype=torch.float32, device=device)
        action = action.argmax(dim=-1).item()

        policy, q_value = net(state)
        policy = policy.view(-1, net.num_outputs)
        q_value = q_value.view(-1, net.num_outputs)

        _, next_q_value = net(next_state)
        next_q_value = next_q_value.view(-1, net.num_outputs)
        next_action = net.get_action(next_state)
        target = reward + mask * gamma * next_q_value[0][next_action]

        log_policy = torch.log(policy[0][action] + 1e-8)
        loss_policy = -log_policy * q_value[0][action].detach()
        loss_value = F.mse_loss(q_value[0][action], target.detach())
        loss = loss_policy + loss_value

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def get_action(self, inputs):
        policy, _ = self.forward(inputs)
        policy = policy.squeeze(0).detach().cpu().numpy()
        return np.random.choice(self.num_outputs, p=policy)


class A2CNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc = nn.Linear(num_inputs, 128)
        self.fc_actor = nn.Linear(128, num_outputs)
        self.fc_critic = nn.Linear(128, 1)
        self.model_name = "Adavantage_Actor_Critic"

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")

    def forward(self, inputs):
        x = F.relu(self.fc(inputs))
        policy = F.softmax(self.fc_actor(x), dim=-1)
        value = self.fc_critic(x)
        return policy, value

    @classmethod
    def train_model(cls, net, transition, optimizer, gamma):
        device = next(net.parameters()).device
        state, next_state, action, reward, mask = transition

        state = state.to(device)
        next_state = next_state.to(device)
        reward = torch.tensor(reward, dtype=torch.float32, device=device)
        mask = torch.tensor(mask, dtype=torch.float32, device=device)
        action = action.argmax(dim=-1)

        policy, value = net(state)
        _, next_value = net(next_state)
        value = value.squeeze()
        next_value = next_value.squeeze()

        target = reward + mask * gamma * next_value
        log_policy = torch.log(policy[0, action] + 1e-8)
        advantage = (target - value).detach()

        loss_actor = -log_policy * advantage
        loss_critic = F.mse_loss(value, target.detach())
        entropy = -(policy * torch.log(policy + 1e-8)).mean()
        loss = loss_actor + loss_critic - 0.05 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def get_action(self, inputs):
        with torch.no_grad():
            policy, _ = self.forward(inputs)
            policy = policy.squeeze(0).detach().cpu().numpy()
            return np.random.choice(self.num_outputs, 1, p=policy)[0]
