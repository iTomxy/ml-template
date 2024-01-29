import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


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
    ref: https://blog.csdn.net/HackerTom/article/details/119740928
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
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(const_mask, np.ndarray):
            const_mask = torch.from_numpy(const_mask).long()
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


class IntermediateLayerGetter:
    """wrap a torch.nn.Module to get its intermediate layer outputs
    From: https://github.com/sebamenabar/Pytorch-IntermediateLayerGetter
    Usage:
        ```python
        getter = IntermediateLayerGetter(network, {
            "<module_name>": "<return_key>",
            ...
        })
        inter_output_dict, final_output = getter(input)
        for return_key, return_value in inter_output_dict.items():
            print(return_key, return_value.size())
        ```
    """

    def __init__(self, model, return_layers):
        """
        model: torch.nn.module, the PyTorch module to call
        return_layers: dict, {<module_name>: <return_key>}
            <module_name> specifies whose output you want to get,
            <return_key> specifies how you want to call this output.
        """
        self._model = model
        self.return_layers = return_layers

    def __call__(self, *args, **kwargs):
        """
        Input:
            (the same as how you call the original module)
        Output:
            ret: OrderedDict, {<return_key>: <return_value>}
                In case a submodule is called more than once, <return_value> will be a list.
            output: tensor, final output
        """
        ret = OrderedDict()
        handles = []
        for name, new_name in self.return_layers.items():
            def hook(module, input, output, new_name=new_name):
                if new_name in ret:
                    if type(ret[new_name]) is list:
                        ret[new_name].append(output)
                    else:
                        ret[new_name] = [ret[new_name], output]
                else:
                    ret[new_name] = output

            try:
                layer = self._model.get_submodule(name)
                h = layer.register_forward_hook(hook)
            except AttributeError as e:
                raise AttributeError(f'Module {name} not found')

            handles.append(h)

        output = self._model(*args, **kwargs)

        for h in handles:
            h.remove() # removes the corresponding added hook

        return ret, output


class UpsampleDeterministic(nn.Module):
    """deterministic upsample with `nearest` interpolation
    From: https://github.com/pytorch/pytorch/issues/12207
    """

    def __init__(self, scale_factor=2):
        """
        Input:
            scale_factor: int or (int, int), ratio to scale (along heigth & width)
        """
        super(UpsampleDeterministic, self).__init__()
        if isinstance(scale_factor, (tuple, list)):
            assert len(scale_factor) == 2
            self.scale_h, self.scale_w = scale_factor
        else:
            self.scale_h = self.scale_w = scale_factor
        assert isinstance(self.scale_h, int) and isinstance(self.scale_w, int)

    def forward(self, x):
        """
        Input:
            x: [n, c, h, w], torch.Tensor
        Output:
            upsampled x': [n, c, h * scale_h, w * scale_w]
        """
        return x[:, :, :, None, :, None].expand(
            -1, -1, -1, self.scale_h, -1, self.scale_w).reshape(
                x.size(0), x.size(1), x.size(2) * self.scale_h, x.size(3) * self.scale_w)


if "__main__" == __name__:
    # build network
    net = nn.Sequential(OrderedDict([
        ('conv', nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.ReLU()
        )),
        ('convt', nn.Sequential(
            nn.ConvTranspose2d(64, 5, 3),
            nn.ReLU()
        ))
    ]))

    # print sub/moduls names <- you get module/layer output by its name
    print(net)
    for module_name, module in net.named_modules():
        print(module_name)

    # get intermedia layer output
    getter = IntermediateLayerGetter(net, {'convt.0': 'feature_map'})

    # forward
    x = torch.ones(1, 28, 28)
    print(x.size())             # [1, 28, 28]
    y = net(x)                  # normal forward
    print(y.size())             # [5, 28, 28]
    ret, y2 = getter(x)         # forward & get intermedia output
    for k, v in ret.items():
        print(k, v.size())      # feature_map [5, 28, 28]
