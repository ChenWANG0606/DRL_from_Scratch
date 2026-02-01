import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RFNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model_name = 'RF' # Policy Gradient

        self.fc_1 = nn.Linear(num_inputs, 128)
        self.fc_2 = nn.Linear(128, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, input):
        x = F.relu(self.fc_1(input))
        policy = F.softmax(self.fc_2(x), dim=-1)
        return policy

    @classmethod
    def train_model(cls, net, transitions, optimizer, gamma):
        device = next(net.parameters()).device
        states, actions, rewards, masks = transitions.state, transitions.action, transitions.reward, transitions.mask

        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        rewards = torch.Tensor(rewards).to(device)
        masks = torch.Tensor(masks).to(device)

        returns = torch.zeros_like(rewards)

        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return * masks[t]
            returns[t] = running_return
        
        policies = net(states)
        policies = policies.view(-1, net.num_outputs)

        log_policies = (torch.log(policies) * actions.detach()).sum(dim=1)

        loss = (-log_policies * returns).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss

    def get_action(self, input):
        policy = self.forward(input)
        policy = policy.squeeze(0).detach().cpu().numpy()

        action = np.random.choice(self.num_outputs, 1, p=policy)[0]
        return action
