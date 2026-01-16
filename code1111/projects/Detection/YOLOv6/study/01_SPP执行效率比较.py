# -*- coding: utf-8 -*-
import time

import torch

from yolov6.layers.common import SimSPPF, CSPSPPFModule, SimCSPSPPF


def calc_param_cnt(net):
    cnt = 0
    for param in net.parameters():
        cnt += param.numel()
    return cnt


def t0():
    c = 12
    x = torch.rand(2, c, 128, 128)
    sppf = SimSPPF(c, c)
    cspsppf = SimCSPSPPF(c, c)
    print(calc_param_cnt(sppf))
    print(calc_param_cnt(cspsppf))
    n = 10

    def c0(_spp):
        print(str(_spp).split("(")[0], "=" * 50)
        st = time.time()
        for i in range(n):
            _spp(x)
        et = time.time()
        print(f"总耗时:{et - st}")

    for i in range(5):
        c0(sppf)
        c0(cspsppf)


if __name__ == '__main__':
    t0()
