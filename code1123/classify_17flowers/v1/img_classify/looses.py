# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/14 20:09
Create User : 19410
Desc : 损失函数相关方法
"""

import torch.nn as nn


def build_losses(**kwargs):
    return nn.CrossEntropyLoss()
