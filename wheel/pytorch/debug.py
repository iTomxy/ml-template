import torch


def check_nan_inf(tensors):
    """tensors: single tensor, or collections of tensor"""
    if isinstance(tensors, torch.Tensor):
        _nan = 1 if torch.isnan(tensors).any() else 0
        _inf = 2 if torch.isinf(tensors).any() else 0
        if _nan or _inf:
            return _nan + _inf
    elif isinstance(tensors, collections.Iterable):
        for _t in tensors:
            res = check_nan_inf(_t)
            if res:
                return res
    else:
        raise Exception("check_nan_inf: unsupported type: {}".format(type(tensors)))
    return 0
