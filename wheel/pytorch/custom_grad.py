import torch


class Htanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, epsilon=1):
        ctx.save_for_backward(x.data, torch.tensor(epsilon))
        return x.sign()

    @staticmethod
    def backward(ctx, dy):
        x, epsilon = ctx.saved_tensors
        dx = torch.where((x < -epsilon) | (x > epsilon), torch.zeros_like(dy), dy)
        return dx, None
