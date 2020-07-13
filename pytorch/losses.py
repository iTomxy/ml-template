import torch
import torch.nn.functional as F


"""Ref:
https://omoindrot.github.io/triplet-loss
https://blog.csdn.net/hustqb/article/details/80361171#commentBox
https://blog.csdn.net/hackertom/article/details/103374313
"""

def struct_loss(X, Y, S, coef=0.5):
    xTy = coef * X.mm(Y.T)
    loss = (1 - S) * xTy - (xTy.sigmoid() + 1e-16).log()
    return loss.mean()
