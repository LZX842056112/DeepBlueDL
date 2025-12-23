# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/9/27 14:01
Create User : 19410
Desc : xxx
"""

import torch
import torch.nn as nn


def tt01():
    bn = nn.BatchNorm2d(num_features=10)

    x = torch.randn((4, 10, 100, 200))
    o = bn(x)
    print(o.shape)

    print("=" * 20)
    for name, param in bn.named_parameters():
        print(name, "--->", param.shape)
    print("=" * 20)
    for name, tensor in bn.state_dict().items():
        print(name, "--->", tensor.shape)


def tt02():
    bn = nn.BatchNorm1d(num_features=5)
    bn.train()
    x = torch.tensor([
        [1, 3, 5, 7, 9],
        [2, 4, 6, 10, 12],
        [1, 9, 2, 8, 7]
    ], dtype=torch.float32)
    o = bn(x)
    print(o)
    print(bn.weight)
    print(bn.bias)
    print(bn.running_mean)
    print(bn.running_var)

    print("=" * 20)
    print(torch.mean(x, dim=0))
    print(torch.std(x, dim=0))
    print(torch.var(x, dim=0))


def tt03():
    ln = nn.LayerNorm(normalized_shape=(3, 10, 20))

    x = torch.randn((4, 3, 10, 20))
    o = ln(x)
    print(o.shape)

    print("=" * 20)
    for name, param in ln.named_parameters():
        print(name, "--->", param.shape)
    print("=" * 20)
    for name, tensor in ln.state_dict().items():
        print(name, "--->", tensor.shape)


def tt04():
    from torchvision import models, transforms, utils

    # 模型迁移/恢复加载
    # 当给定weights=models.VGG16_BN_Weights.DEFAULT或者pretrained=True的时候，就会从网络上下载训练好的模型参数，并完成参数恢复操作
    # 下载好的模型默认会放到当用户根目录下: ~\.cache\torch\hub\checkpoints；如果想更改默认的保存路径，可以考虑配置一个环境变量: XDG_CACHE_HOME
    # 下载好的模型(vgg16)默认为基于ImageNet数据集训练的一个通用开源数据集
    # ImageNet总共有1000个类别: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a#file-imagenet1000_clsid_to_human-txt
    net = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
    print(net)

    print("=" * 100)
    bn = net.features[1]
    print(bn)
    print(bn.weight)
    print(bn.bias)
    print(bn.running_mean)
    print(bn.running_var)

if __name__ == '__main__':
    tt04()
