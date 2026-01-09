# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/30 20:12
Create User : 19410
Desc : xxx
"""
import time

import torch
from torchvision import models
from thop import profile


def t0():
    net = models.vgg16()
    x = torch.randn(1, 3, 224, 224)
    r = net(x)
    print(r.shape)

    param_cnt = 0
    for param in net.parameters():
        param_cnt += param.numel()
    print(f"参数量为: {param_cnt}")

    # 执行两次
    for i in range(2):
        net(x)

    n = 20
    xn = [torch.rand(1, 3, 224, 224) for _ in range(n)]
    st = time.time()
    for xi in xn:
        net(xi)
    et = time.time()
    print(f"单次运行平均耗时:{(et - st) / n:.5f}s")


def t1():
    # net = models.vgg16()
    # net = models.shufflenet_v2_x2_0()
    net = models.detection.fasterrcnn_mobilenet_v3_large_fpn()
    net.eval()
    x = torch.randn(1, 3, 224, 224)
    total_ops, total_params = profile(net, (x,))
    print(type(net))
    print(f"总浮点运算量:{total_ops / 1000 / 1000 / 1000:.3f}GFlops")
    print(f"总参数量:{total_params / 1000 / 1000:.2f}M")

    # 执行两次
    for i in range(2):
        net(x)

    n = 20
    xn = [torch.rand(1, 3, 224, 224) for _ in range(n)]
    st = time.time()
    for xi in xn:
        net(xi)
    et = time.time()
    print(f"单次运行平均耗时:{(et - st) / n:.5f}s")


if __name__ == '__main__':
    t1()
