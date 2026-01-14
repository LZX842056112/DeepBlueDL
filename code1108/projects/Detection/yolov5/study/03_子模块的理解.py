# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/2 11:54
Create User : 19410
Desc : xxx
"""

import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
print(sys.path)

import torch

from models.common import *


def t1():
    pool5 = nn.MaxPool2d(5, 1, padding=2)
    pool9 = nn.MaxPool2d(9, 1, padding=4)
    pool13 = nn.MaxPool2d(13, 1, padding=6)

    x = torch.rand(4, 3, 26, 26)
    n = 1000

    t1 = time.time()
    for i in range(n):
        r5 = pool5(x)
        r9 = pool5(r5)
        r13 = pool5(r9)
        sppf_r = torch.cat([
            x, r5, r9, r13
        ], dim=1)
    t2 = time.time()

    for i in range(n):
        spp_r = torch.cat([
            x, pool5(x), pool9(x), pool13(x)
        ], dim=1)
    t3 = time.time()

    print(f"耗时: {t3 - t2} --- {t2 - t1}")
    print(torch.max(torch.abs(sppf_r - spp_r)))


def t2():
    conv = Conv(3, 64, k=6, s=2, p=2)

    x = torch.randn(4, 3, 320, 320)
    r = conv(x)
    print(r.shape)


def t3():
    c3 = C3(128, 128, n=1, e=0.2)
    print(c3)

    x = torch.randn(4, 128, 80, 80)
    r = c3(x)
    print(r.shape)


def t4():
    c3 = C3m(128, 128, n=2)
    print(c3)
    x = torch.randn(4, 128, 80, 80)
    r = c3(x)
    print(r.shape)


def t5():
    mp = MP(128)
    x = torch.randn(4, 128, 80, 80)
    r = mp(x)
    print(r.shape)

if __name__ == '__main__':
    t5()
