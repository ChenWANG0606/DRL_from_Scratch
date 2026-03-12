import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.common import NoisyLinear


class Noisy_DQN(nn.Module):
    def __init__(self, num_inputs, num_actions, sequence_length):
        super().__init__()
        self.model_name = "Noisy_DQN"
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.sequence_length = sequence_length
        self.fc1 = nn.Linear(num_inputs * sequence_length, 128)
        self.fc2 = NoisyLinear(128, num_actions)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = x.view(-1, self.num_inputs * self.sequence_length)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def reset_noise(self):
        if isinstance(self.fc1, NoisyLinear):
            self.fc1.reset_noise()
        if isinstance(self.fc2, NoisyLinear):
            self.fc2.reset_noise()

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, gamma):
        device = next(online_net.parameters()).device

        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        actions = torch.tensor(np.array(batch.action), dtype=torch.float32, device=device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).view(-1)
        masks = torch.tensor(batch.mask, dtype=torch.float32, device=device).view(-1)

        online_net.reset_noise()
        target_net.reset_noise()

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
