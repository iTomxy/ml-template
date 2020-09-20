import torch
import torch.nn.functional as F


def l2_regularize(w_list):
    loss = 0
    for w in w_list:
        loss += 0.5 * (w ** 2).sum()
    return loss
