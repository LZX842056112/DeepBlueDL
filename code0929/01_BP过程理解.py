# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/9/29 20:41
Create User : 19410
Desc : BP过程理解
"""

import numpy as np

_w = np.asarray([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
_b = np.asarray([0.35, 0.65])

# 假定就一条样本
_x = np.asarray([5.0, 10.0])
_y = np.asarray([0.01, 0.99])

lr = 0.5


def w(i):
    return _w[i - 1]


def b(i):
    return _b[i - 1]


def x(i):
    return _x[i - 1]


def y(i):
    return _y[i - 1]


def update_w(i, gd):
    global _w, lr
    _i = i - 1
    _w[_i] = _w[_i] - lr * gd


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def training():
    # 1. 前向过程 FP -- 模型的前向预测结果 + 损失
    h1 = sigmoid(z=w(1) * x(1) + w(2) * x(2) + b(1))
    h2 = sigmoid(z=w(3) * x(1) + w(4) * x(2) + b(1))
    h3 = sigmoid(z=w(5) * x(1) + w(6) * x(2) + b(1))
    o1 = sigmoid(z=w(7) * h1 + w(9) * h2 + w(11) * h3 + b(2))
    o2 = sigmoid(z=w(8) * h1 + w(10) * h2 + w(12) * h3 + b(2))
    # print(o1, o2)
    loss = 0.5 * (y(1) - o1) ** 2 + 0.5 * (y(2) - o2) ** 2
    # print(loss)

    # 2. 反向过程 BP -- 基于loss求解梯度 + 基于梯度更新参数
    gd_loss_o1 = o1 - y(1)  # 损失关于Out_o1的导数
    gd_loss_o2 = o2 - y(2)  # 损失关于Out_o2的导数

    # gd_loss_w7 = gd_loss_o1 * o1 * (1 - o1) * h1  # 损失关于w7的导数
    # print(gd_loss_w7)
    # update_w(7, gd=gd_loss_w7)
    # print(_w)

    # 按照PPT的理解过程写的代码
    # update_w(7, gd=gd_loss_o1 * o1 * (1 - o1) * h1)
    # update_w(8, gd=gd_loss_o2 * o2 * (1 - o2) * h1)
    # update_w(9, gd=gd_loss_o1 * o1 * (1 - o1) * h2)
    # update_w(10, gd=gd_loss_o2 * o2 * (1 - o2) * h2)
    # update_w(11, gd=gd_loss_o1 * o1 * (1 - o1) * h3)
    # update_w(12, gd=gd_loss_o2 * o2 * (1 - o2) * h3)
    #
    # update_w(1, gd=(gd_loss_o1 * o1 * (1 - o1) * w(7) + gd_loss_o2 * o2 * (1 - o2) * w(8)) * h1 * (1 - h1) * x(1))
    # update_w(2, gd=(gd_loss_o1 * o1 * (1 - o1) * w(7) + gd_loss_o2 * o2 * (1 - o2) * w(8)) * h1 * (1 - h1) * x(2))
    # update_w(3, gd=(gd_loss_o1 * o1 * (1 - o1) * w(9) + gd_loss_o2 * o2 * (1 - o2) * w(10)) * h2 * (1 - h2) * x(1))
    # update_w(4, gd=(gd_loss_o1 * o1 * (1 - o1) * w(9) + gd_loss_o2 * o2 * (1 - o2) * w(10)) * h2 * (1 - h2) * x(2))
    # update_w(5, gd=(gd_loss_o1 * o1 * (1 - o1) * w(11) + gd_loss_o2 * o2 * (1 - o2) * w(12)) * h3 * (1 - h3) * x(1))
    # update_w(6, gd=(gd_loss_o1 * o1 * (1 - o1) * w(11) + gd_loss_o2 * o2 * (1 - o2) * w(12)) * h3 * (1 - h3) * x(2))

    gds = [
        (gd_loss_o1 * o1 * (1 - o1) * w(7) + gd_loss_o2 * o2 * (1 - o2) * w(8)) * h1 * (1 - h1) * x(1),
        (gd_loss_o1 * o1 * (1 - o1) * w(7) + gd_loss_o2 * o2 * (1 - o2) * w(8)) * h1 * (1 - h1) * x(2),
        (gd_loss_o1 * o1 * (1 - o1) * w(9) + gd_loss_o2 * o2 * (1 - o2) * w(10)) * h2 * (1 - h2) * x(1),
        (gd_loss_o1 * o1 * (1 - o1) * w(9) + gd_loss_o2 * o2 * (1 - o2) * w(10)) * h2 * (1 - h2) * x(2),
        (gd_loss_o1 * o1 * (1 - o1) * w(11) + gd_loss_o2 * o2 * (1 - o2) * w(12)) * h3 * (1 - h3) * x(1),
        (gd_loss_o1 * o1 * (1 - o1) * w(11) + gd_loss_o2 * o2 * (1 - o2) * w(12)) * h3 * (1 - h3) * x(2),
        gd_loss_o1 * o1 * (1 - o1) * h1,
        gd_loss_o2 * o2 * (1 - o2) * h1,
        gd_loss_o1 * o1 * (1 - o1) * h2,
        gd_loss_o2 * o2 * (1 - o2) * h2,
        gd_loss_o1 * o1 * (1 - o1) * h3,
        gd_loss_o2 * o2 * (1 - o2) * h3
    ]
    for _i in range(len(gds)):
        update_w(
            i=_i + 1,
            gd=gds[_i]
        )

    return (loss, o1, o2)


if __name__ == '__main__':
    print(_w)
    r = training()
    print(_w)
    print(r)
    print("=" * 100)
    for _j in range(1000):
        _r = training()
    print(_r)
    print(_w)
