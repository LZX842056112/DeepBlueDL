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
