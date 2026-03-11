import torch

class TNPGOptimizer:

    def __init__(self, net, lr=0.05):
        self.net = net
        self.lr = lr

    @staticmethod
    def flat_grad(grads):
        return torch.cat([g.contiguous().view(-1) for g in grads])

    @staticmethod
    def flat_params(model):
        return torch.cat([p.data.view(-1) for p in model.parameters()])

    @staticmethod
    def update_model(model, new_params):
        index = 0
        for param in model.parameters():
            param_len = param.numel()
            param.data.copy_(new_params[index:index+param_len].view(param.size()))
            index += param_len

    def step(self, loss, step_dir):
        grads = torch.autograd.grad(loss, self.net.parameters())
        loss_grad = self.flat_grad(grads).detach()

        params = self.flat_params(self.net)
        new_params = params - self.lr * step_dir

        self.update_model(self.net, new_params)

        return loss_grad