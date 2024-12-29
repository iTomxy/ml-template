import numpy as np
import torch


def get_on(*tensors, cuda=True):
    """np.ndarray -> torch.Tensor (cuda)"""
    res = []
    for t in tensors:
        if isinstance(t, (int, float)):
            t = torch.tensor(t)
            if cuda:
                t = t.cuda()
        elif isinstance(t, np.ndarray):
            t = torch.from_numpy(t)
            if cuda:
                t = t.cuda()
        elif isinstance(t, (list, tuple)):
            _res = get_on(*t, cuda=cuda)
            if isinstance(_res, list):
                # res.extend(_res)
                res.append(_res)  # as a list as it was
            else:
                res.append([_res])  # as a list as it was
        elif isinstance(t, dict):
            res.append({k: get_on(v, cuda=cuda) for k, v in t.items()})
        else:
            assert isinstance(t, torch.Tensor)
            if cuda and "cuda" not in t.device.type:
                t = t.cuda()
            res.append(t)
    if len(res) == 1:
        res = res[0]
    return res


def get_off(*tensors):
    """torch.Tensor (cuda) -> np.ndarray"""
    res = []
    for t in tensors:
        if isinstance(t, (int, float, str, np.ndarray)):
            res.append(t)
        elif isinstance(t, (list, tuple)):
            _res = get_off(*t)
            if isinstance(_res, list):
                # res.extend(_res)
                res.append(_res)  # as a list as it was
            else:
                res.append([_res])  # as a list as it was
        elif isinstance(t, dict):
            res.append({k: get_off(v) for k, v in t.items()})
        else:
            assert isinstance(t, torch.Tensor)
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


def gpus_type():
    """detect types of each GPU based on PyTorch
    NOTE: this method is limited by `CUDA_VISIBLE_DEVICES`,
    i.e. it only shows GPUs that visible to PyTorch.
    """
    if torch.cuda.is_available():
        gpu_types = {i: torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())}
    else:
        gpu_types = {}

    print("GPU types:", gpu_types)
    return gpu_types
