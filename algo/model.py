import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import NoisyLinear

class Reinforce(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model_name = 'Reinforce' # Policy Gradient Reinforce

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
    

# advantage(average baseline estimation + returns-based weight)
class Reinforce2(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model_name = 'Reinforce-baseline' # Policy Gradient Reinforce

        self.fc_1 = nn.Linear(num_inputs, 128)
        self.fc_2 = nn.Linear(128, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    def forward(self, input):
        x = F.relu(self.fc_1(input))
        policy = F.softmax(self.fc_2(x), dim=-1)
        return policy
    @staticmethod
    def get_ae(rewards, masks, gamma):
        returns = torch.zeros_like(rewards)
        
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return * masks[t]
            returns[t] = running_return

        baseline = returns.mean()
        advantages = returns - baseline 
        return returns, advantages
        
    @classmethod
    def train_model(cls, net, transitions, optimizer, gamma):
        device = next(net.parameters()).device
        states, actions, rewards, masks = transitions.state, transitions.action, transitions.reward, transitions.mask

        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        rewards = torch.Tensor(rewards).to(device)
        masks = torch.Tensor(masks).to(device)

        returns, advantages = cls.get_ae(rewards, masks, gamma)
        
        policies = net(states)
        policies = policies.view(-1, net.num_outputs)

        log_policies = (torch.log(policies) * actions.detach()).sum(dim=1)

        loss = (-log_policies * advantages).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss

    def get_action(self, input):
        policy = self.forward(input)
        policy = policy.squeeze(0).detach().cpu().numpy()

        action = np.random.choice(self.num_outputs, 1, p=policy)[0]
        return action





# TODO PPO1
# class PPO1(nn.Module):
#     def __init__(self, num_inputs, num_actions):
#         super().__init__()
#         self.num_inputs = num_inputs
#         self.num_actions = num_actions
#         self.fc = nn.Linear(num_inputs, 128)
#         self.model_name = 'PPO' # Proximal Policy Optimization

        
#         self.fc_1 = nn.Linear(num_inputs, 128)
#         self.fc_2 = nn.Linear(128, num_actions)

#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')


#     def forward(self, input):
#         x = F.relu(self.fc_1(input))
#         policy = F.softmax(self.fc_2(x), dim=-1)
#         return policy
    
#     def train_model(cls, net, transitions, optimizer, gamma):
#         pass

#     def get_action(self, input):
#         policy = self.forward(input)
#         policy = policy.squeeze(0).detach().cpu().numpy()
#         action = np.random.choice(self.num_outputs, 1, p=policy)[0]
#         return action

# TODO PPO2
# class PPO2(nn.Module):





class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions, sequence_length):
        super().__init__()
        self.model_name = "DQN"
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.sequence_length = sequence_length
        self.fc1 = nn.Linear(num_inputs*sequence_length, 128)
        self.fc2 = nn.Linear(128, num_actions)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        x = x.view(-1, self.num_inputs * self.sequence_length)
        x = F.relu(self.fc1(x))
        qvalue = self.fc2(x)
        return qvalue

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, gamma):
        device = next(online_net.parameters()).device

        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        actions = torch.tensor(np.array(batch.action), dtype=torch.float32, device=device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).view(-1)
        masks   = torch.tensor(batch.mask, dtype=torch.float32, device=device).view(-1)

        q_values = online_net(states)
        with torch.no_grad():
            target_value = target_net(next_states).max(dim=1)[0]

        # 按照Q值公式计算选择动作a后每个经验的Q值，并累加
        # Q_\pi(s_t,a_t)
        # dim = 1代表最后的形状是dim1形状，也就是将dim0求和，最后剩下一列数据
        Q_pred = torch.sum(q_values*actions,dim=1)
        # y = r_i+\underset{a}{max}\hat{Q}(s_{i+1},a)
        target = rewards + masks*gamma*target_value

        loss = F.mse_loss(Q_pred, target)
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
        self.fc1 = nn.Linear(num_inputs*sequence_length, 128)
        self.fc2 = nn.Linear(128, num_actions)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        x = x.view(-1, self.num_inputs * self.sequence_length)
        x = F.relu(self.fc1(x))
        qvalue = self.fc2(x)
        return qvalue

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, gamma):
        device = next(online_net.parameters()).device

        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        actions = torch.tensor(np.array(batch.action), dtype=torch.float32, device=device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).view(-1)
        masks   = torch.tensor(batch.mask, dtype=torch.float32, device=device).view(-1)

        q_values = online_net(states)
        with torch.no_grad():
            # online network 选择动作
            next_actions = online_net(next_states).argmax(dim=1)

            # target network 计算Q值
            target_q = target_net(next_states)
            target_value = target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        # 按照Q值公式计算选择动作a后每个经验的Q值，并累加
        # Q_\pi(s_t,a_t)
        # dim = 1代表最后的形状是dim1形状，也就是将dim0求和，最后剩下一列数据
        Q_pred = torch.sum(q_values*actions,dim=1)
        # y = r_i+\underset{a}{max}\hat{Q}(s_{i+1},a)
        target = rewards + masks*gamma*target_value

        loss = F.mse_loss(Q_pred, target)
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
        self.fc1 = nn.Linear(num_inputs*sequence_length, 128)
        self.fcV = nn.Linear(128, 1)
        self.fcA = nn.Linear(128, num_actions)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        x = x.view(-1, self.num_inputs * self.sequence_length)
        x = F.relu(self.fc1(x))
        V = self.fcV(x)
        A = self.fcA(x)
        A_mean = A.mean(dim=1, keepdim=True)
        qvalue = V + (A - A_mean)
        return qvalue

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, gamma):
        device = next(online_net.parameters()).device

        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        actions = torch.tensor(np.array(batch.action), dtype=torch.float32, device=device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).view(-1)
        masks   = torch.tensor(batch.mask, dtype=torch.float32, device=device).view(-1)

        q_values = online_net(states)
        with torch.no_grad():
            # online network 选择动作
            next_actions = online_net(next_states).argmax(dim=1)

            # target network 计算Q值
            target_q = target_net(next_states)
            target_value = target_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        # 按照Q值公式计算选择动作a后每个经验的Q值，并累加
        # Q_\pi(s_t,a_t)
        # dim = 1代表最后的形状是dim1形状，也就是将dim0求和，最后剩下一列数据
        Q_pred = torch.sum(q_values*actions,dim=1)
        # y = r_i+\underset{a}{max}\hat{Q}(s_{i+1},a)
        target = rewards + masks*gamma*target_value

        loss = F.mse_loss(Q_pred, target)
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
        self.fc1 = nn.Linear(num_inputs*sequence_length, 128)
        self.fc2 = nn.Linear(128, num_actions)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        x = x.view(-1, self.num_inputs * self.sequence_length)
        x = F.relu(self.fc1(x))
        qvalue = self.fc2(x)
        return qvalue

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, gamma, n_step):
        device = next(online_net.parameters()).device

        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        actions = torch.tensor(np.array(batch.action), dtype=torch.float32, device=device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).view(-1)
        masks   = torch.tensor(batch.mask, dtype=torch.float32, device=device).view(-1)

        q_values = online_net(states)
        with torch.no_grad():
            target_value = target_net(next_states).max(dim=1)[0]

        # 按照Q值公式计算选择动作a后每个经验的Q值，并累加
        # Q_\pi(s_t,a_t)
        # dim = 1代表最后的形状是dim1形状，也就是将dim0求和，最后剩下一列数据
        Q_pred = torch.sum(q_values*actions,dim=1)
        # y = r_i+\underset{a}{max}\hat{Q}(s_{i+1},a)
        target = rewards + masks*(gamma**n_step)*target_value

        loss = F.mse_loss(Q_pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss
    
    def get_action(self, inputs):
        with torch.no_grad():
            qvalue = self.forward(inputs)
            _, action = torch.max(qvalue, 1)
        return action.detach().cpu().numpy().squeeze()
    
class PER_DQN(nn.Module):
    def __init__(self, num_inputs, num_actions, sequence_length):
        super().__init__()
        self.model_name = "PER"
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.sequence_length = sequence_length
        self.fc1 = nn.Linear(num_inputs*sequence_length, 128)
        self.fc2 = nn.Linear(128, num_actions)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        x = x.view(-1, self.num_inputs * self.sequence_length)
        x = F.relu(self.fc1(x))
        qvalue = self.fc2(x)
        return qvalue
    def get_td_error(online_net, target_net, batch, gamma):
        device = next(online_net.parameters()).device

        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        actions = torch.tensor(np.array(batch.action), dtype=torch.float32, device=device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).view(-1)
        masks   = torch.tensor(batch.mask, dtype=torch.float32, device=device).view(-1)

        q_values = online_net(states)
        with torch.no_grad():
            target_value = target_net(next_states).max(dim=1)[0]

        # 按照Q值公式计算选择动作a后每个经验的Q值，并累加
        # Q_\pi(s_t,a_t)
        # dim = 1代表最后的形状是dim1形状，也就是将dim0求和，最后剩下一列数据
        Q_pred = torch.sum(q_values*actions,dim=1)
        # y = r_i+\underset{a}{max}\hat{Q}(s_{i+1},a)
        target = rewards + masks*gamma*target_value
        td_error = Q_pred - target
        return td_error


    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, gamma, weights):
        td_error = cls.get_td_error(online_net, target_net, batch, gamma)
        loss = torch.mean(td_error**2*weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, td_error.detach().cpu().numpy()
    
    def get_action(self, inputs):
        with torch.no_grad():
            qvalue = self.forward(inputs)
            _, action = torch.max(qvalue, 1)
        return action.detach().cpu().numpy().squeeze()    

class Noisy_DQN(nn.Module):
    def __init__(self, num_inputs, num_actions, sequence_length):
        super().__init__()
        self.model_name = "Noisy_DQN"
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.sequence_length = sequence_length
        self.fc1 = nn.Linear(num_inputs*sequence_length, 128)
        self.fc2 = NoisyLinear(128, num_actions)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        x = x.view(-1, self.num_inputs * self.sequence_length)
        x = F.relu(self.fc1(x))
        qvalue = self.fc2(x)
        return qvalue
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
        masks   = torch.tensor(batch.mask, dtype=torch.float32, device=device).view(-1)

        online_net.reset_noise()
        target_net.reset_noise()

        q_values = online_net(states)
        with torch.no_grad():
            target_value = target_net(next_states).max(dim=1)[0]

        # 按照Q值公式计算选择动作a后每个经验的Q值，并累加
        # Q_\pi(s_t,a_t)
        # dim = 1代表最后的形状是dim1形状，也就是将dim0求和，最后剩下一列数据
        Q_pred = torch.sum(q_values*actions,dim=1)
        # y = r_i+\underset{a}{max}\hat{Q}(s_{i+1},a)
        target = rewards + masks*gamma*target_value

        loss = F.mse_loss(Q_pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss
    
    def get_action(self, inputs):
        with torch.no_grad():
            qvalue = self.forward(inputs)
            _, action = torch.max(qvalue, 1)
        return action.detach().cpu().numpy().squeeze()    

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
        self.support = torch.linspace(vmin, vmax, atoms)# value that might be taken after discretization
        self.fc1 = nn.Linear(num_inputs*sequence_length, 128)
        self.fc2 = nn.Linear(128, num_actions * atoms)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        x = x.view(-1, self.num_inputs * self.sequence_length)
        x = F.relu(self.fc1(x))
        dist = self.fc2(x)
        dist = dist.view(-1, self.num_actions, self.atoms)
        dist = F.softmax(dist, dim = 2)
        return dist
    
    def _get_q(self, dist):
        support = self.support.to(dist.device)
        q = torch.sum(dist * support, dim=2)
        return q
    
    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, gamma):
        device = next(online_net.parameters()).device

        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        actions = torch.tensor(np.array(batch.action), dtype=torch.float32, device=device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).view(-1,1)
        masks   = torch.tensor(batch.mask, dtype=torch.float32, device=device).view(-1,1)

        atoms = online_net.atoms
        vmin = online_net.vmin
        vmax = online_net.vmax
        delta_z = online_net.delta_z
        support = online_net.support.to(device)

        dist = online_net(states)
        action_index = actions.argmax(dim=1)
        dist = dist[range(dist.size(0)), action_index]

        with torch.no_grad():
            next_dist = target_net(next_states)
            # 新的分布和原本分布不在一个空间，先选择动作的分布

            next_q = torch.sum(next_dist * support, dim=2)# 算出期望
            next_action = next_q.argmax(dim=1)# 求期望最大分布的动作
            # 整数编号，因此要创建第一位用来索引
            
            next_dist = next_dist[range(next_dist.size(0)), next_action]
            # 对范围进行裁剪，因为要通过分桶来映射回去，不能超出分桶
            Tz = rewards + masks * gamma * support
            Tz = Tz.clamp(vmin, vmax)

            b = (Tz - vmin) / delta_z
            l = b.floor().long()
            u = b.ceil().long()
            target_dist = torch.zeros_like(next_dist)

            # for i in range(atoms):
            #     target_dist[range(target_dist.size(0)), l[:, i]] += next_dist[:, i] * (u[:, i] - b[:, i])
            #     target_dist[range(target_dist.size(0)), u[:, i]] += next_dist[:, i] * (b[:, i] - l[:, i])
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
            qvalue  = self._get_q(dist)
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

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

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
            l = b.floor().long()
            u = b.ceil().long()

            l = l.clamp_(0, online_net.atoms - 1)
            u = u.clamp_(0, online_net.atoms - 1)

            target_dist = torch.zeros_like(next_target_dist)
            offset = torch.arange(target_dist.size(0), device=device).unsqueeze(1)

            same = (u == l)
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ACNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc = nn.Linear(num_inputs, 128)

        # actor
        self.fc_actor = nn.Linear(128, num_outputs)

        # critic (输出每个动作Q)
        self.fc_critic = nn.Linear(128, num_outputs)

        self.model_name = "One_Step_Actor_Critic"#训不动这个模型

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

        action = action.argmax(dim=-1).item()

        policy, q_value = net(state)
        policy = policy.view(-1, net.num_outputs)
        q_value = q_value.view(-1, net.num_outputs)

        _, next_q_value = net(next_state)
        next_q_value = next_q_value.view(-1, net.num_outputs)

        # 用策略采样 next action
        next_action = net.get_action(next_state)

        # TD target
        target = reward + mask * gamma * next_q_value[0][next_action]

        # Actor loss
        log_policy = torch.log(policy[0][action] + 1e-8)
        loss_policy = -log_policy * q_value[0][action].detach()

        # Critic loss
        loss_value = F.mse_loss(q_value[0][action], target.detach())

        loss = loss_policy + loss_value

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss


    def get_action(self, input):
        policy, _ = self.forward(input)
        policy = policy.squeeze(0).detach().cpu().numpy()
        action = np.random.choice(self.num_outputs, p=policy)
        return action



class A2CNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc = nn.Linear(num_inputs, 128)
        self.fc_actor = nn.Linear(128, num_outputs)
        self.fc_critic = nn.Linear(128, 1)
        self.model_name = "Adavantage_Actor_Critic"

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

    def forward(self, input):
        x = F.relu(self.fc(input))
        policy = F.softmax(self.fc_actor(x), dim=-1)
        v = self.fc_critic(x)
        return policy, v

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

        log_policy = torch.log(policy[0, action]+1e-8)
        advantage = (target - value).detach()

        loss_actor = -log_policy * advantage
        loss_critic = F.mse_loss(value, target.detach())
        entropy = -(policy * torch.log(policy + 1e-8)).mean()

        loss = loss_actor + loss_critic - 0.05*entropy# 系数不能太大，否则干扰训练

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, input):
        with torch.no_grad():
            policy, _ = self.forward(input)
            policy = policy.squeeze(0).detach().cpu().numpy()

            action = np.random.choice(self.num_outputs, 1, p=policy)[0]
        return action
    

class GAE(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc = nn.Linear(num_inputs, 128)
        self.fc_actor = nn.Linear(128, num_outputs)
        self.fc_critic = nn.Linear(128, 1)
        self.model_name = "GAE"

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

    def forward(self, input):
        x = F.relu(self.fc(input))
        policy = F.softmax(self.fc_actor(x), dim=-1)
        v = self.fc_critic(x)
        return policy, v
    
    def get_gae(self, values, next_values, rewards, masks, gamma, lambda_gae):
        advantages = torch.zeros_like(rewards)
        running_advantage = 0.0

        for t in range(len(rewards) - 1, -1, -1):
            delta = rewards[t] + gamma * next_values[t] * masks[t] - values[t]
            running_advantage = delta + gamma * lambda_gae * running_advantage * masks[t]
            advantages[t] = running_advantage

        returns = advantages + values
        return returns, advantages
    
    @classmethod
    def train_model(cls, net, optimizer, batch, gamma, lambda_gae, critic_coefficient, entropy_coefficient):
        device = next(net.parameters()).device

        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        actions = torch.stack(batch.action).to(device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).view(-1)
        masks   = torch.tensor(batch.mask, dtype=torch.float32, device=device).view(-1)

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

        entropy = -(policies * torch.log(policies + 1e-8)).sum(dim=1).mean()# 对整个整个batch求平均

        loss = actor_loss + critic_coefficient * critic_loss - entropy_coefficient * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss
    
    def get_action(self, input):
        with torch.no_grad():
            policy, _ = self.forward(input)
            policy = policy.squeeze(0).detach().cpu().numpy()

            action = np.random.choice(self.num_outputs, 1, p=policy)[0]
        return action


import copy
class TNPG(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model_name = "TNPG"

        self.fc_1 = nn.Linear(num_inputs, 128)
        self.fc_2 = nn.Linear(128, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

    def forward(self, input):
        x = F.relu(self.fc_1(input))
        policy = F.softmax(self.fc_2(x), dim=-1)
        return policy

    # flatten gradient
    @staticmethod
    def flat_grad(grads):
        grad_flatten = []
        for grad in grads:
            grad_flatten.append(grad.contiguous().view(-1))
        grad_flatten = torch.cat(grad_flatten)
        return grad_flatten

    # KL divergence
    @staticmethod
    def kl_divergence(net, old_net, states):
        policy = net(states)
        old_policy = old_net(states).detach()

        kl = old_policy * torch.log((old_policy + 1e-8) / (policy + 1e-8))
        kl = kl.sum(1)
        return kl.mean()

    # flatten hessian
    @staticmethod
    def flat_hessian(hessians):
        hessians_flatten = []
        for hessian in hessians:
            hessians_flatten.append(hessian.contiguous().view(-1))
        hessians_flatten = torch.cat(hessians_flatten)
        return hessians_flatten

    # Fisher Vector Product
    @staticmethod
    def fisher_vector_product(net, states, p, cg_damp=0.1):

        old_net = copy.deepcopy(net)

        kl = TNPG.kl_divergence(net, old_net, states)

        kl_grad = torch.autograd.grad(
            kl, net.parameters(), create_graph=True
        )

        kl_grad = TNPG.flat_grad(kl_grad)

        kl_grad_p = (kl_grad * p).sum()

        kl_hessian_p = torch.autograd.grad(
            kl_grad_p, net.parameters(), retain_graph=True
        )

        kl_hessian_p = TNPG.flat_hessian(kl_hessian_p)

        return kl_hessian_p + cg_damp * p

    # flatten params
    @staticmethod
    def flat_params(model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))
        return torch.cat(params)

    # update model params
    @staticmethod
    def update_model(model, new_params):

        index = 0
        for param in model.parameters():

            param_length = param.numel()

            new_param = new_params[index:index + param_length]

            param.data.copy_(new_param.view(param.size()))

            index += param_length

    # Conjugate Gradient
    @staticmethod
    def conjugate_gradient(net, states, b, n_step=10, residual_tol=1e-10, cg_damp = 0.1):

        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()

        r_dot_r = torch.dot(r, r)

        for _ in range(n_step):

            Avp = TNPG.fisher_vector_product(net, states, p, cg_damp)

            alpha = r_dot_r / (torch.dot(p, Avp) + 1e-8)

            x += alpha * p

            r -= alpha * Avp

            new_r_dot_r = torch.dot(r, r)

            if new_r_dot_r < residual_tol:
                break

            beta = new_r_dot_r / r_dot_r

            p = r + beta * p

            r_dot_r = new_r_dot_r

        return x

    @classmethod
    def train_model(cls, net, transitions, gamma, lr=1.0, max_kl = 0.01, cg_damp = 0.1, cg_iters = 10):

        device = next(net.parameters()).device

        states = torch.stack(transitions.state).to(device)
        actions = torch.stack(transitions.action).to(device)
        rewards = torch.tensor(transitions.reward, dtype=torch.float32).to(device)
        masks = torch.tensor(transitions.mask, dtype=torch.float32).to(device)

        returns = torch.zeros_like(rewards)

        running_return = 0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return * masks[t]
            returns[t] = running_return

        
        advantages = returns - returns.mean()
        advantages = advantages / (advantages.std(unbiased=False) + 1e-8)

        policies = net(states)
        policies = policies.view(-1, net.num_outputs)

        log_policies = (torch.log(policies+1e-8) * actions.detach()).sum(dim=1)

        objective = (log_policies * advantages.detach()).sum()

        # policy gradient
        grads = torch.autograd.grad(objective, net.parameters())
        loss_grad = cls.flat_grad(grads).detach()

        # conjugate gradient solve
        step_dir = cls.conjugate_gradient(net, states, loss_grad, n_step=cg_iters, cg_damp=cg_damp)
        fisher_step = cls.fisher_vector_product(net, states, step_dir, cg_damp=cg_damp)
        step_norm = torch.dot(step_dir, fisher_step)
        step_scale = torch.sqrt(
            torch.tensor(2.0 * max_kl, device=device) / (step_norm + 1e-8)
        )
        full_step = step_scale * step_dir
        # update params
        params = cls.flat_params(net)

        new_params = params + lr * full_step

        cls.update_model(net, new_params)

        return -objective

    def get_action(self, input):

        policy = self.forward(input)

        policy = policy.squeeze(0).detach().cpu().numpy()

        action = np.random.choice(self.num_outputs, 1, p=policy)[0]

        return action
