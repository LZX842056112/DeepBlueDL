# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/14 20:11
Create User : 19410
Desc : 优化器创建相关工具方法
"""
from dataclasses import dataclass

import torch.nn as nn

import torch.optim as optim


@dataclass
class OptimArgs:
    net: nn.Module
    lr: float
    weight_decay: float = 0.0


def build_optim(args: OptimArgs):
    return optim.SGD(params=args.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def build_optim_v2(args: OptimArgs):
    net = args.net
    if not hasattr(net, 'vgg'):
        return build_optim(args)

    g0, g1 = [], []
    for name, param in net.named_parameters():
        if name.startswith("vgg."):
            g0.append(param)
        else:
            g1.append(param)

    params = [
        {
            'params': g0, 'lr': args.lr * 0.1,
        },  # 针对迁移模块的参数
        {'params': g1, 'lr': args.lr}  # 针对非迁移模块的参数
    ]
    return optim.SGD(params=params, lr=args.lr, weight_decay=args.weight_decay)
