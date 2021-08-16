import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class TrainVar(nn.Module):
    """wrapper of a trainable variable"""

    def __init__(self, *size, init_val=None, process_fn=None):
        """
        Input:
            size: d1, ..., dn
            init_val: constant initializer
            process_fn: something to do before returning the var,
                e.g. normalization, activation, etc.
        """
        super(TrainVar, self).__init__()
        self.size = size
        self.process_fn = process_fn
        if init_val is None:
            self.weight = Parameter(torch.Tensor(*size))
            self.reset_parameters()
        else:
            self.weight = Parameter(init_val * torch.ones(*size, dtype=torch.float))

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self):
        if self.process_fn is None:
            return self.weight
        else:
            return self.process_fn(self.weight)

    def extra_repr(self):
        return 'size={}'.format(self.size)


class MixVar(nn.Module):
    """mixture of constants & trainable variables
    - https://blog.csdn.net/HackerTom/article/details/119740928
    """

    def __init__(self, X, const_mask, init_val=None, process_fn=None):
        """
        Input:
            X: [n, d], FULL matrix including both constants & (placeholders of) variables
            const_mask: [n], in {0, 1}, indicating whether the i-th item is constant
            init_val: constant initializer of variables
            process_fn: something to do before returning the var,
                e.g. normalization, activation, etc.
        """
        super(MixVar, self).__init__()
        self.X = X
        self.const_mask = const_mask
        self.process_fn = process_fn
        self.full_indices = np.arange(X.size(0))

        assert X.size(0) == const_mask.size(0)
        n = X.size(0)
        n_const = const_mask.sum()
        n_var = n - n_const
        assert n_var > 0, "* constant only, no need to use this class"
        size = [n_var, X.size(1)]

        if init_val is None:
            self.weight = Parameter(torch.Tensor(*size))
            self.reset_parameters()
        else:
            self.weight = Parameter(init_val * torch.ones(*size, dtype=torch.float))

        # map the full id in `X` to the relative one in `weight`
        _cnt = 0
        self.id_map = {}
        for i in range(n):
            if 0 == const_mask[i]:
                self.id_map[i] = _cnt
                _cnt += 1
        assert _cnt == n_var

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, index=None):
        """MUST use this function for slicing instead of slicing manually"""
        if index is None:
            index = self.full_indices
        res = torch.zeros(index.shape[0], self.X.size(1),
            dtype=self.X.dtype).to(self.weight.device)
        for i in range(index.shape[0]):
            _idx = index[i]
            if self.const_mask[_idx] > 0:
                res[i] = self.X[_idx].to(self.weight.device)
            else:
                res[i] = self.weight[self.id_map[_idx]]

        if self.process_fn:
            res = self.process_fn(res)
        return res

    def extra_repr(self):
        return 'size={}'.format(self.X.size())
