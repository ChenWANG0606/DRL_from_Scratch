import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions, sequence_length):
        super().__init__()
        self.model_name = "DQN"
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.sequence_length = sequence_length
        self.fc1 = nn.Linear(num_inputs * sequence_length, 128)
        self.fc2 = nn.Linear(128, num_actions)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = x.view(-1, self.num_inputs * self.sequence_length)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, gamma):
        device = next(online_net.parameters()).device

        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        actions = torch.tensor(np.array(batch.action), dtype=torch.float32, device=device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).view(-1)
        masks = torch.tensor(batch.mask, dtype=torch.float32, device=device).view(-1)

        q_values = online_net(states)
        with torch.no_grad():
            target_value = target_net(next_states).max(dim=1)[0]

        q_pred = torch.sum(q_values * actions, dim=1)
        target = rewards + masks * gamma * target_value

        loss = F.mse_loss(q_pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def get_action(self, inputs):
        with torch.no_grad():
            qvalue = self.forward(inputs)
            _, action = torch.max(qvalue, 1)
        return action.detach().cpu().numpy().squeeze()


class DDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, sequence_length):
        super().__init__()
        self.model_name = "DDQN"
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.sequence_length = sequence_length
        self.fc1 = nn.Linear(num_inputs * sequence_length, 128)
        self.fc2 = nn.Linear(128, num_actions)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = x.view(-1, self.num_inputs * self.sequence_length)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, gamma):
        device = next(online_net.parameters()).device

        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        actions = torch.tensor(np.array(batch.action), dtype=torch.float32, device=device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).view(-1)
        masks = torch.tensor(batch.mask, dtype=torch.float32, device=device).view(-1)

        q_values = online_net(states)
        with torch.no_grad():
            next_actions = online_net(next_states).argmax(dim=1)
            target_q = target_net(next_states)
            target_value = target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        q_pred = torch.sum(q_values * actions, dim=1)
        target = rewards + masks * gamma * target_value

        loss = F.mse_loss(q_pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def get_action(self, inputs):
        with torch.no_grad():
            qvalue = self.forward(inputs)
            _, action = torch.max(qvalue, 1)
        return action.detach().cpu().numpy().squeeze()


class D3QN(nn.Module):
    def __init__(self, num_inputs, num_actions, sequence_length):
        super().__init__()
        self.model_name = "Dueling DDQN"
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.sequence_length = sequence_length
        self.fc1 = nn.Linear(num_inputs * sequence_length, 128)
        self.fc_v = nn.Linear(128, 1)
        self.fc_a = nn.Linear(128, num_actions)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = x.view(-1, self.num_inputs * self.sequence_length)
        x = F.relu(self.fc1(x))
        value = self.fc_v(x)
        advantage = self.fc_a(x)
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        return value + (advantage - advantage_mean)

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, gamma):
        device = next(online_net.parameters()).device

        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        actions = torch.tensor(np.array(batch.action), dtype=torch.float32, device=device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).view(-1)
        masks = torch.tensor(batch.mask, dtype=torch.float32, device=device).view(-1)

        q_values = online_net(states)
        with torch.no_grad():
            next_actions = online_net(next_states).argmax(dim=1)
            target_q = target_net(next_states)
            target_value = target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        q_pred = torch.sum(q_values * actions, dim=1)
        target = rewards + masks * gamma * target_value

        loss = F.mse_loss(q_pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def get_action(self, inputs):
        with torch.no_grad():
            qvalue = self.forward(inputs)
            _, action = torch.max(qvalue, 1)
        return action.detach().cpu().numpy().squeeze()


class multistep_DQN(nn.Module):
    def __init__(self, num_inputs, num_actions, sequence_length):
        super().__init__()
        self.model_name = "multistep_DQN"
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.sequence_length = sequence_length
        self.fc1 = nn.Linear(num_inputs * sequence_length, 128)
        self.fc2 = nn.Linear(128, num_actions)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = x.view(-1, self.num_inputs * self.sequence_length)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, gamma, n_step):
        device = next(online_net.parameters()).device

        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        actions = torch.tensor(np.array(batch.action), dtype=torch.float32, device=device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).view(-1)
        masks = torch.tensor(batch.mask, dtype=torch.float32, device=device).view(-1)

        q_values = online_net(states)
        with torch.no_grad():
            target_value = target_net(next_states).max(dim=1)[0]

        q_pred = torch.sum(q_values * actions, dim=1)
        target = rewards + masks * (gamma ** n_step) * target_value

        loss = F.mse_loss(q_pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def get_action(self, inputs):
        with torch.no_grad():
            qvalue = self.forward(inputs)
            _, action = torch.max(qvalue, 1)
        return action.detach().cpu().numpy().squeeze()
