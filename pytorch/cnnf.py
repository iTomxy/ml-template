import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


"""Ref:
- https://zhuanlan.zhihu.com/p/29786939
- https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
- https://pytorch.org/docs/stable/nn.html#zeropad2d
- https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/pad.md
"""


class CNN_F(nn.Module):
    """CNN-F / VGG-F"""

    def __init__(self, weight_file):
        super(CNN_F, self).__init__()
        layers = sio.loadmat(weight_file)["net"][0][0][0][0]
        # print(layers.shape)  # (19,)
        self.first = nn.Sequential(
            make_conv(layers[0]),
            nn.ReLU(inplace=True),
            make_lrn(layers[2]),
            make_pool(layers[3])
        )
        self.second = nn.Sequential(
            make_conv(layers[4]),
            nn.ReLU(inplace=True),
            make_lrn(layers[6]),
            make_pool(layers[7])
        )
        self.third = nn.Sequential(
            make_conv(layers[8]),
            nn.ReLU(inplace=True)
        )
        self.fourth = nn.Sequential(
            make_conv(layers[10]),
            nn.ReLU(inplace=True)
        )
        self.fifth = nn.Sequential(
            make_conv(layers[12]),
            nn.ReLU(inplace=True),
            make_pool(layers[14])
        )
        self.sixth = nn.Sequential(
            make_conv(layers[15]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.seventh = nn.Sequential(
            make_conv(layers[17]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        x = self.third(x)
        x = self.fourth(x)
        x = self.fifth(x)
        x = self.sixth(x)
        x = self.seventh(x)
        return x.view(x.size(0), -1)  # (n, 4096)


def make_conv(layer):
    """pytorch: (n, C, h, w)
    tf: (n, h, w, C)
    ref:
    - https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L329
    """
    layer = layer[0][0]
    # print("name:", layer[0])
    # print("type:", layer[1])
    k, b = layer[2][0]
    #b = b.flatten()
    # print("kernel:", k.shape, ", bias:", b.shape)
    shape = layer[3][0]
    # print("shape:", shape)
    pad = layer[4][0]
    # print("pad:", pad)
    stride = layer[5][0]
    # print("stride:", stride)
    
    conv = nn.Conv2d(shape[2], shape[3], shape[:2],
                     stride=tuple(stride))  # must convert to tuple
                    #  padding=tuple(pad))
    conv.weight.data = torch.from_numpy(k.transpose((3, 2, 0, 1)))
    conv.bias.data = torch.from_numpy(b.flatten())
    
    if np.sum(pad) > 0:
        padding = nn.ZeroPad2d(tuple(pad.astype(np.int32)))
        conv = nn.Sequential(padding, conv)

    return conv


class LRN(nn.Module):
    """ref:
    - https://zhuanlan.zhihu.com/p/29786939
    - https://www.jianshu.com/p/c06aea337d5d
    """
    def __init__(self, local_size=1, bias=1.0, alpha=1.0, beta=0.75, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1), #0.2.0_4会报错，需要在最新的分支上AvgPool3d才有padding参数
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0)) 
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))

        self.bias = bias
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(self.bias).pow(self.beta)#这里的1.0即为bias
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(self.bias).pow(self.beta)
        x = x.div(div)
        return x


def make_lrn(layer):
    layer = layer[0][0]
    # print("name:", layer[0])
    # print("type:", layer[1])
    param = layer[2][0]
    # print("local_size/depth_radius:", param[0])
    # print("bias:", param[1])
    # print("alpha:", param[2])
    # print("beta:", param[3])
    
    lrn = LRN(int(param[0]), param[1], param[2], param[3])
    return lrn


def make_pool(layer):
    layer = layer[0][0]
    # print("name:", layer[0])
    # print("type:", layer[1])
    # print("pool type:", layer[2])
    k_size = layer[3][0]
    stride = layer[4][0]
    # print("stride:", stride)
    pad = layer[5][0]
    # print("pad:", pad)

    pool = nn.MaxPool2d(tuple(k_size),
                        stride=tuple(stride))
                        # padding=tuple(pad))
    if np.sum(pad) > 0:
        padding = nn.ZeroPad2d(tuple(pad.astype(np.int32)))
        pool = nn.Sequential(padding, pool)

    return pool


if __name__ == "__main__":
    cnnf = CNN_F("E:/iTom/dataset/CNN-F/vgg_net.mat")
    print(cnnf)
    x = torch.empty(1, 3, 224, 224)
    o = cnnf(x)
    print(o.size())
