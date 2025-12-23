# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/9/27 15:51
Create User : 19410
Desc : xxx
"""

import torch
import torch.nn as nn


def t0():
    drop_prob = 0.1
    # 参数p表示的针对每个样本而言，将多少百分比的特征drop为0
    dropout = nn.Dropout(p=drop_prob)
    dropout.train()
    # dropout.eval()

    # 训练的时候，随机对部分(占比为p)输出值重置为0，并且将其它(1-p)的部分输出值进行更改（乘以缩放系数： 1/(1-p)）
    # 推理的时候，对所有输出值保持原样输出

    x = torch.randn(2, 10)  # 上一个模块的输出
    o = dropout(x)
    print("=" * 10)
    print(x)
    print(o)
    print(o / (1 / (1 - drop_prob)))


def t1():
    p = 0.2
    dropout = nn.Dropout(p=p)  # p就是有多少特征dropout为0
    x1 = torch.rand(2, 1, 6, 6)  # 一般是全连接或者激活操作之后的输出特征向量
    print(x1)

    # https://zhuanlan.zhihu.com/p/410309087
    # 训练的时候
    dropout.train()
    z1 = dropout(x1)
    print(z1 / x1)  # 1 / (1-p)
    print(z1)
    print("=" * 20)
    # print(dropout(x1))  # 相当于第二个批次的dropout，和第一次相比，会出现不同位置重置为0
    print(f"原始数据均值 {torch.mean(x1)}")
    mask = (z1 > 0.0).to(x1.dtype)
    x2 = x1 * mask  # 相当于仅做重置为0，但是不对保留的特征做处理
    print(x2)
    print(f"只做drop的均值 {torch.mean(x2)}")
    x3 = x2 * (1 / (1 - p))  # 相当于对保留的特征值做一个缩放的处理
    print(x3)
    print(f"只做drop的均值 {torch.mean(x3)}")
    print("=" * 20)

    # 推理的时候
    dropout.eval()
    z2 = dropout(x1)
    print(z2)


def t2():
    dropout = nn.Dropout2d(p=0.5)
    x = torch.randn(2, 10, 4, 4)
    o = dropout(x)
    print(x)
    print("=" * 20)
    print(o)


# DropBlock: 如果需要在卷积之后使用dropout类似的操作，建议DropBlock

if __name__ == '__main__':
    t0()


