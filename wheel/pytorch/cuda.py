import numpy as np
import torch


def get_on(*tensors, cuda=True):
    """np.ndarray -> torch.Tensor (cuda)"""
    if cuda:
        fn = lambda x: torch.from_numpy(x).cuda()
    else:
        fn = lambda x: torch.from_numpy(x)
    res = list(map(fn, tensors))
    if len(res) == 1:
        res = res[0]
    return res


def get_off(*tensors):
    """torch.Tensor (cuda) -> np.ndarray"""
    res = []
    for t in tensors:
        if isinstance(t, (int, float, str, np.ndarray)):
            res.append(t)
            continue
        t = t.data
        if "cuda" in t.device.type:
            t = t.cpu()
        if 0 == t.ndim:  # scalar
            t = t.item()
        else:
            t = t.numpy()
        res.append(t)
    if len(res) == 1:
        res = res[0]
    return res
