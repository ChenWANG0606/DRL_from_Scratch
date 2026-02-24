import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RFNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model_name = 'RF' # Policy Gradient Reinforce

        self.fc_1 = nn.Linear(num_inputs, 128)
        self.fc_2 = nn.Linear(128, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
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

# todo still can't be trained stably
class ACNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc = nn.Linear(num_inputs, 128)
        self.fc_actor = nn.Linear(128, num_outputs)
        self.fc_critic = nn.Linear(128, num_outputs)
        self.model_name = "Actor_Critic"

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

    def forward(self, input):
        x = F.relu(self.fc(input))
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
        action = action.argmax(dim=-1)

        policy, q_value = net(state)
        policy, q_value = policy.view(-1, net.num_outputs), q_value.view(-1, net.num_outputs)
        _, next_q_value = net(next_state)
        next_q_value = next_q_value.view(-1, net.num_outputs)
        next_action = net.get_action(next_state)


        target = reward + mask * gamma * next_q_value[0][next_action]

        log_policy = torch.log(policy[0, action])
        advantage = (target - q_value[0, action]).detach()

        loss_policy = -log_policy * advantage
        loss_value = F.mse_loss(q_value[0, action], target.detach())

        loss = loss_policy + loss_value
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, input):
        policy, _ = self.forward(input)
        policy = policy.squeeze(0).detach().cpu().numpy()

        action = np.random.choice(self.num_outputs, 1, p=policy)[0]
        return action