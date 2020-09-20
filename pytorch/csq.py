import torch
import torch.nn as nn
import torchvision
from args import *


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hash_bit = args.bit
        self.base_model = getattr(torchvision.models, args.model_type)(pretrained=True)

        self.conv1 = self.base_model.conv1
        self.bn1 = self.base_model.bn1
        self.relu = self.base_model.relu
        self.maxpool = self.base_model.maxpool
        self.layer1 = self.base_model.layer1
        self.layer2 = self.base_model.layer2
        self.layer3 = self.base_model.layer3
        self.layer4 = self.base_model.layer4
        self.avgpool = self.base_model.avgpool
        self.feature_layers = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.maxpool,
            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.fc1 = nn.Linear(self.base_model.fc.in_features, self.base_model.fc.in_features)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(self.base_model.fc.in_features, self.base_model.fc.in_features)
        self.activation2 = nn.ReLU()
        self.fc3 = nn.Linear(self.base_model.fc.in_features, self.hash_bit)
        self.last_layer = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.hash_layer = nn.Sequential(
            self.fc1, self.activation1, self.dropout, self.fc2, self.activation2,
            self.fc3, self.last_layer)

        self.iter_num = 0
        self.scale = 1

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        #y = self.last_layer(5*y)
        return y


class AlexNetFc(nn.Module):
    def __init__(self):
        super(AlexNetFc, self).__init__()
        self.base_model = torchvision.models.alexnet(pretrained=True)
        self.features = self.base_model.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), self.base_model.classifier[i])
        self.feature_layers = nn.Sequential(self.features, self.classifier)

        self.hash_bit = args.bit
        feature_dim = self.base_model.classifier[6].in_features
        self.fc1 = nn.Linear(feature_dim, feature_dim)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(feature_dim, feature_dim)
        self.activation2 = nn.ReLU()
        self.fc3 = nn.Linear(feature_dim, self.hash_bit)
        self.last_layer = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.hash_layer = nn.Sequential(self.fc1, self.activation1, self.fc2, self.activation2, self.fc3,
                                        self.last_layer)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        y = self.hash_layer(x)

        return y


def single_centre(C, L):
    idx = (L == 1).nonzero()[:, 1]
    return C[idx]


def multi_centre(C, L, C_random):
    c_ls = []
    for l in L:
        idx = (l == 1).nonzero().squeeze(1)
        cs = C[idx].mean(0)
        cs[cs < 0] = -1
        cs[cs > 0] = 1
        idx = (cs == 0)
        cs[idx] = C_random[idx]
        c_ls.append(cs.view(1, -1))  # [1, bit]

    Cs = torch.cat(c_ls)  # [n, bit]
    return Cs  # {-1, 1}


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.7 ** (epoch // 10))
    #for param_group in optimizer.param_groups:
        #param_group['lr'] = lr
    optimizer.param_groups[0]['lr'] = args.lrp * lr
    optimizer.param_groups[1]['lr'] = lr

    return lr
