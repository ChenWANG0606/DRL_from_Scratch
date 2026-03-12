import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TNPG(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model_name = "TNPG"

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
    def flat_grad(grads):
        return torch.cat([grad.contiguous().view(-1) for grad in grads])

    @staticmethod
    def kl_divergence(policy, old_policy):
        kl = old_policy * torch.log((old_policy + 1e-8) / (policy + 1e-8))
        return kl.sum(1, keepdim=True)

    @staticmethod
    def flat_hessian(hessians):
        return torch.cat([hessian.contiguous().view(-1) for hessian in hessians])

    @staticmethod
    def fisher_vector_product(net, states, p, cg_damp=0.1):
        policy = net(states)
        old_policy = net(states).detach()

        kl = TNPG.kl_divergence(policy, old_policy).mean()
        kl_grad = torch.autograd.grad(kl, net.parameters(), create_graph=True)
        kl_grad = TNPG.flat_grad(kl_grad)
        kl_grad_p = (kl_grad * p).sum()
        kl_hessian_p = torch.autograd.grad(kl_grad_p, net.parameters(), retain_graph=True)
        kl_hessian_p = TNPG.flat_hessian(kl_hessian_p)
        return kl_hessian_p + cg_damp * p

    @staticmethod
    def flat_params(model):
        return torch.cat([param.data.view(-1) for param in model.parameters()])

    @staticmethod
    def update_model(model, new_params):
        index = 0
        for param in model.parameters():
            param_length = param.numel()
            new_param = new_params[index : index + param_length]
            param.data.copy_(new_param.view(param.size()))
            index += param_length

    @staticmethod
    def conjugate_gradient(net, states, b, n_step=10, residual_tol=1e-10, cg_damp=0.1):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        r_dot_r = torch.dot(r, r)

        for _ in range(n_step):
            avp = TNPG.fisher_vector_product(net, states, p, cg_damp)
            alpha = r_dot_r / (torch.dot(p, avp) + 1e-8)
            x += alpha * p
            r -= alpha * avp
            new_r_dot_r = torch.dot(r, r)
            if new_r_dot_r < residual_tol:
                break
            beta = new_r_dot_r / r_dot_r
            p = r + beta * p
            r_dot_r = new_r_dot_r

        return x

    @classmethod
    def train_model(cls, net, transitions, gamma, lr=1.0, max_kl=0.01, cg_damp=0.1, cg_iters=10):
        device = next(net.parameters()).device

        states = torch.stack(transitions.state).to(device)
        actions = torch.stack(transitions.action).to(device)
        rewards = torch.tensor(transitions.reward, dtype=torch.float32, device=device)
        masks = torch.tensor(transitions.mask, dtype=torch.float32, device=device)

        returns = torch.zeros_like(rewards)
        running_return = 0
        for index in reversed(range(len(rewards))):
            running_return = rewards[index] + gamma * running_return * masks[index]
            returns[index] = running_return

        advantages = returns - returns.mean()
        advantages = advantages / (advantages.std(unbiased=False) + 1e-8)

        policies = net(states).view(-1, net.num_outputs)
        log_policies = (torch.log(policies + 1e-8) * actions.detach()).sum(dim=1)
        objective = (log_policies * advantages.detach()).sum()

        grads = torch.autograd.grad(objective, net.parameters())
        loss_grad = cls.flat_grad(grads).detach()

        step_dir = cls.conjugate_gradient(net, states, loss_grad, n_step=cg_iters, cg_damp=cg_damp)
        fisher_step = cls.fisher_vector_product(net, states, step_dir, cg_damp=cg_damp)
        step_norm = torch.dot(step_dir, fisher_step)
        step_scale = torch.sqrt(torch.tensor(2.0 * max_kl, device=device) / (step_norm + 1e-8))
        full_step = step_scale * step_dir
        params = cls.flat_params(net)
        new_params = params + full_step
        cls.update_model(net, new_params)
        return -objective

    def get_action(self, inputs):
        policy = self.forward(inputs)
        policy = policy.squeeze(0).detach().cpu().numpy()
        return np.random.choice(self.num_outputs, 1, p=policy)[0]


class TRPO(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model_name = "TRPO"

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
    def flat_grad(grads):
        return torch.cat([grad.contiguous().view(-1) for grad in grads])

    @staticmethod
    def kl_divergence(policy, old_policy):
        kl = old_policy * torch.log((old_policy + 1e-8) / (policy + 1e-8))
        return kl.sum(1, keepdim=True)

    @staticmethod
    def flat_hessian(hessians):
        return torch.cat([hessian.contiguous().view(-1) for hessian in hessians])

    @staticmethod
    def fisher_vector_product(net, states, p, cg_damp=0.1):
        policy = net(states)
        old_policy = net(states).detach()
        kl = TNPG.kl_divergence(policy, old_policy).mean()
        kl_grad = torch.autograd.grad(kl, net.parameters(), create_graph=True)
        kl_grad = TNPG.flat_grad(kl_grad)
        kl_grad_p = (kl_grad * p).sum()
        kl_hessian_p = torch.autograd.grad(kl_grad_p, net.parameters(), retain_graph=True)
        kl_hessian_p = TNPG.flat_hessian(kl_hessian_p)
        return kl_hessian_p + cg_damp * p

    @staticmethod
    def flat_params(model):
        return torch.cat([param.data.view(-1) for param in model.parameters()])

    @staticmethod
    def update_model(model, new_params):
        index = 0
        for param in model.parameters():
            param_length = param.numel()
            new_param = new_params[index : index + param_length]
            param.data.copy_(new_param.view(param.size()))
            index += param_length

    @staticmethod
    def conjugate_gradient(net, states, b, n_step=10, residual_tol=1e-10, cg_damp=0.1):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        r_dot_r = torch.dot(r, r)

        for _ in range(n_step):
            avp = TNPG.fisher_vector_product(net, states, p, cg_damp)
            alpha = r_dot_r / (torch.dot(p, avp) + 1e-8)
            x += alpha * p
            r -= alpha * avp
            new_r_dot_r = torch.dot(r, r)
            if new_r_dot_r < residual_tol:
                break
            beta = new_r_dot_r / r_dot_r
            p = r + beta * p
            r_dot_r = new_r_dot_r

        return x

    @classmethod
    def train_model(cls, net, transitions, gamma, max_kl=0.01, cg_damp=0.1, cg_iters=10):
        device = next(net.parameters()).device

        states = torch.stack(transitions.state).to(device)
        actions = torch.stack(transitions.action).to(device)
        rewards = torch.tensor(transitions.reward, dtype=torch.float32, device=device)
        masks = torch.tensor(transitions.mask, dtype=torch.float32, device=device)

        returns = torch.zeros_like(rewards)
        running_return = 0
        for index in reversed(range(len(rewards))):
            running_return = rewards[index] + gamma * running_return * masks[index]
            returns[index] = running_return

        advantages = returns - returns.mean()
        advantages = advantages / (advantages.std(unbiased=False) + 1e-8)

        policies = net(states).view(-1, net.num_outputs)
        log_policies = (torch.log(policies + 1e-8) * actions.detach()).sum(dim=1)

        old_policies = net(states).detach().view(-1, net.num_outputs)
        old_log_policies = (torch.log(old_policies + 1e-8) * actions.detach()).sum(dim=1)
        ratio = torch.exp(log_policies - old_log_policies)
        surrogate_objective = (ratio * advantages.detach()).sum()

        surrogate_loss_grads = torch.autograd.grad(surrogate_objective, net.parameters())
        surrogate_loss_grads = cls.flat_grad(surrogate_loss_grads).detach()
        params = cls.flat_params(net)

        step_dir = cls.conjugate_gradient(net, states, surrogate_loss_grads, n_step=cg_iters, cg_damp=cg_damp)
        fisher_step = cls.fisher_vector_product(net, states, step_dir, cg_damp=cg_damp)
        with torch.no_grad():
            step_norm = torch.dot(step_dir, fisher_step)
            step_scale = torch.sqrt(torch.tensor(2.0 * max_kl, device=device) / (step_norm + 1e-8))
            full_step = step_scale * step_dir

            fraction = 1.0
            for _ in range(10):
                new_params = params + fraction * full_step
                cls.update_model(net, new_params)

                policies = net(states).view(-1, net.num_outputs)
                log_policies = (torch.log(policies + 1e-8) * actions).sum(dim=1)
                ratio = torch.exp(log_policies - old_log_policies)
                surrogate_objective = (ratio * advantages).mean()

                kl = cls.kl_divergence(policies, old_policies).mean()
                if kl < max_kl:
                    break
                fraction *= 0.5

            return -surrogate_objective

    def get_action(self, inputs):
        policy = self.forward(inputs)
        policy = policy.squeeze(0).detach().cpu().numpy()
        return np.random.choice(self.num_outputs, 1, p=policy)[0]
