import torch
import torch.nn.functional as F


def struct_loss(X, Y, S, coef=0.5):
    xTy = coef * X.mm(Y.T)
    loss = (1 - S) * xTy - (xTy.sigmoid() + 1e-16).log()
    return loss.mean()
