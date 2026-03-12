import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.common import NoisyLinear


class C51(nn.Module):
    def __init__(self, num_inputs, num_actions, sequence_length, atoms=51, vmin=-10, vmax=10):
        super().__init__()
        self.model_name = "Distributional_DQN_C51"
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.sequence_length = sequence_length
        self.atoms = atoms
        self.vmin = vmin
        self.vmax = vmax
        self.delta_z = (vmax - vmin) / (atoms - 1)
        self.support = torch.linspace(vmin, vmax, atoms)
        self.fc1 = nn.Linear(num_inputs * sequence_length, 128)
        self.fc2 = nn.Linear(128, num_actions * atoms)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = x.view(-1, self.num_inputs * self.sequence_length)
        x = F.relu(self.fc1(x))
        dist = self.fc2(x).view(-1, self.num_actions, self.atoms)
        return F.softmax(dist, dim=2)

    def _get_q(self, dist):
        support = self.support.to(dist.device)
        return torch.sum(dist * support, dim=2)

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, gamma):
        device = next(online_net.parameters()).device

        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        actions = torch.tensor(np.array(batch.action), dtype=torch.float32, device=device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).view(-1, 1)
        masks = torch.tensor(batch.mask, dtype=torch.float32, device=device).view(-1, 1)

        dist = online_net(states)
        action_index = actions.argmax(dim=1)
        dist = dist[range(dist.size(0)), action_index]

        with torch.no_grad():
            next_dist = target_net(next_states)
            support = online_net.support.to(device)
            next_q = torch.sum(next_dist * support, dim=2)
            next_action = next_q.argmax(dim=1)
            next_dist = next_dist[range(next_dist.size(0)), next_action]

            tz = rewards + masks * gamma * support
            tz = tz.clamp(online_net.vmin, online_net.vmax)
            b = (tz - online_net.vmin) / online_net.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            target_dist = torch.zeros_like(next_dist)

            target_dist.scatter_add_(1, l, next_dist * (u.float() - b))
            target_dist.scatter_add_(1, u, next_dist * (b - l.float()))

        loss = -(target_dist * torch.log(dist + 1e-8)).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def get_action(self, inputs):
        with torch.no_grad():
            dist = self.forward(inputs)
            qvalue = self._get_q(dist)
            _, action = torch.max(qvalue, 1)
        return action.detach().cpu().numpy().squeeze()


class Rainbow(nn.Module):
    def __init__(self, num_inputs, num_actions, sequence_length, atoms=51, vmin=-10, vmax=10):
        super().__init__()
        self.model_name = "Rainbow"
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.sequence_length = sequence_length
        self.atoms = atoms
        self.vmin = vmin
        self.vmax = vmax
        self.delta_z = (vmax - vmin) / (atoms - 1)
        self.register_buffer("support", torch.linspace(vmin, vmax, atoms))

        self.fc1 = nn.Linear(num_inputs * sequence_length, 128)
        self.value_hidden = NoisyLinear(128, 128)
        self.value_out = NoisyLinear(128, atoms)
        self.adv_hidden = NoisyLinear(128, 128)
        self.adv_out = NoisyLinear(128, num_actions * atoms)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = x.view(-1, self.num_inputs * self.sequence_length)
        x = F.relu(self.fc1(x))

        value = F.relu(self.value_hidden(x))
        value = self.value_out(value).view(-1, 1, self.atoms)

        advantage = F.relu(self.adv_hidden(x))
        advantage = self.adv_out(advantage).view(-1, self.num_actions, self.atoms)

        dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.softmax(dist, dim=2)

    def reset_noise(self):
        self.value_hidden.reset_noise()
        self.value_out.reset_noise()
        self.adv_hidden.reset_noise()
        self.adv_out.reset_noise()

    def _get_q(self, dist):
        return torch.sum(dist * self.support, dim=2)

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, gamma, n_step=1, weights=None):
        device = next(online_net.parameters()).device

        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        actions = torch.tensor(np.array(batch.action), dtype=torch.float32, device=device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).view(-1, 1)
        masks = torch.tensor(batch.mask, dtype=torch.float32, device=device).view(-1, 1)

        online_net.reset_noise()
        target_net.reset_noise()

        dist = online_net(states)
        action_index = actions.argmax(dim=1)
        dist = dist[torch.arange(dist.size(0), device=device), action_index]

        with torch.no_grad():
            next_online_dist = online_net(next_states)
            next_action = online_net._get_q(next_online_dist).argmax(dim=1)

            next_target_dist = target_net(next_states)
            next_target_dist = next_target_dist[torch.arange(next_target_dist.size(0), device=device), next_action]

            support = online_net.support
            tz = rewards + masks * (gamma ** n_step) * support.view(1, -1)
            tz = tz.clamp(online_net.vmin, online_net.vmax)

            b = (tz - online_net.vmin) / online_net.delta_z
            l = b.floor().long().clamp_(0, online_net.atoms - 1)
            u = b.ceil().long().clamp_(0, online_net.atoms - 1)

            target_dist = torch.zeros_like(next_target_dist)
            offset = torch.arange(target_dist.size(0), device=device).unsqueeze(1)
            same = u == l
            target_dist[offset, l] += next_target_dist * same.float()
            target_dist.scatter_add_(1, l, next_target_dist * (~same).float() * (u.float() - b))
            target_dist.scatter_add_(1, u, next_target_dist * (~same).float() * (b - l.float()))

        per_sample_loss = -(target_dist * torch.log(dist + 1e-8)).sum(dim=1)
        if weights is not None:
            loss = torch.mean(per_sample_loss * weights.to(device))
        else:
            loss = per_sample_loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if weights is not None:
            return loss, per_sample_loss.detach().cpu().numpy()
        return loss

    def get_action(self, inputs):
        with torch.no_grad():
            dist = self.forward(inputs)
            qvalue = self._get_q(dist)
            _, action = torch.max(qvalue, 1)
        return action.detach().cpu().numpy().squeeze()
