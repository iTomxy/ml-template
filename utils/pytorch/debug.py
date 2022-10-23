import collections
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


def replace_nan_inf(tensor, v=0):
    """replace inf & nan in `tensor` with `v`"""
    if 0 == v:
        z = torch.zeros_like(tensor)
    else:
        z = torch.ones_like(tensor)
        if 1 != v:
            z = z * v
    return torch.where(torch.isinf(tensor) | torch.isnan(tensor), z, tensor)


def forward_nan(m, x, y):
    if not isinstance(x, tuple):
        x = [x]
    if not isinstance(y, tuple):
        y = [y]
    for i, _x in enumerate(x):
        if isinstance(_x, torch.Tensor) and torch.isnan(_x).any():
            # print("{} output NaN".format(m.__class__.__name__))
            raise RuntimeError("{}: {}-th input NaN".format(m.__class__.__name__, i))
    for i, _y in enumerate(y):
        if isinstance(_y, torch.Tensor) and torch.isnan(_y).any():
            # print("{} output NaN".format(m.__class__.__name__))
            raise RuntimeError("{}: {}-th output NaN".format(m.__class__.__name__, i))


def backward_nan(m, dy, dy_dx):
    if not isinstance(dy, tuple):
        dy = [dy]
    if not isinstance(dy_dx, tuple):
        dy_dx = [dy_dx]
    for i, _dy in enumerate(dy):
        if isinstance(_dy, torch.Tensor):
            # _pn = _dy.grad
            # if _pn is not None:
            #     _pn = _pn.data.norm(2).cpu().item()
            # print("{} grad {}:".format(m.__class__.__name__, i), _pn)
            if torch.isnan(_dy).any():
                # print("{} output NaN".format(m.__class__.__name__))
                raise RuntimeError("{}: {}-th input grad dy NaN".format(m.__class__.__name__, i))
    for i, _dy_dx in enumerate(dy_dx):
        if isinstance(_dy_dx, torch.Tensor) and torch.isnan(_dy_dx).any():
            # print("{} output NaN".format(m.__class__.__name__))
            raise RuntimeError("{}: {}-th output grad dy NaN".format(m.__class__.__name__, i))


def hook_it(model):
    for _m in model.modules():
        # if _m.__class__.__name__ not in ("TALR", "AlexNet", "Sequential"):
        # print(_m.__class__.__name__)
        # _m.register_forward_hook(forward_nan)
        # _m.register_backward_hook(backward_nan)
        name_list = []
        for name, p in _m.named_parameters():
            name_list.append(name)
        print(_m.__class__.__name__, ':', name_list)


def show_model(module, layer=0):
    for name, m in module.named_children():
        if layer > 0:
            print("|  " * (layer - 1) + "|- ", end="")
        print(m.__class__)
        if isinstance(m, nn.Sequential):
            go(m, layer + 1)
