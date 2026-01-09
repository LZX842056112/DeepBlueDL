# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/28 20:31
Create User : 19410
Desc : xxx
"""
import torch


def t0():
    from network import VGG

    net = VGG(
        num_classes=10,
        # layers=[
        #     64, 64, 'M',
        #     128, 128, 'M',
        #     256, 256, 256, 256, 'M',
        #     512, 512, 512, 512, 'M',
        #     512, 512, 512, 512, 'M'
        # ]
        layers=[
            64, 64, 'M',
            128, 128, 'M',
            256, 256, ('C', 256, 1), 'M',
            512, 512, ('C', 512, 1), 'M',
            512, 512, ('C', 512, 1), 'M'
        ]
    )
    print(net)
    x = torch.randn(
        4, 3, 244, 244
    )
    print(x.shape)
    r = net(x)
    print(r.shape)

    # 将网络进行静态转换，一般情况下转换目的为：模型部署以及可视化查看结构
    # https://netron.app/
    jit_net = torch.jit.trace(net, x)
    torch.jit.save(jit_net, "vgg.pt")


def t1():
    from network import GoogLeNet

    net = GoogLeNet(
        num_classes=10
    )
    print(net)
    x = torch.randn(
        4, 3, 224, 224
    )
    print(x.shape)
    r = net(x)
    print(r.shape)

    # 将网络进行静态转换，一般情况下转换目的为：模型部署以及可视化查看结构
    # https://netron.app/
    jit_net = torch.jit.trace(net.eval(), x)
    torch.jit.save(jit_net, "googlenet.pt")


def t2():
    from network import ResNet

    net = ResNet(
        num_classes=10,
        has_se=True
    )
    print(net)
    x = torch.randn(
        4, 3, 224, 224
    )
    print(x.shape)
    r = net(x)
    print(r.shape)

    # 将网络进行静态转换，一般情况下转换目的为：模型部署以及可视化查看结构
    # https://netron.app/
    jit_net = torch.jit.trace(net.eval(), x)
    torch.jit.save(jit_net, "resnet.pt")


def t3():
    from torchvision import models

    # net = models.resnet152()
    net = models.shufflenet_v2_x1_0()
    print(net)
    x = torch.randn(4, 3, 224, 224)
    r = net(x)
    print(r.shape)

    # 将网络进行静态转换，一般情况下转换目的为：模型部署以及可视化查看结构
    # https://netron.app/
    jit_net = torch.jit.trace(net.eval(), x)
    torch.jit.save(jit_net, "torchvision_model.pt")


if __name__ == '__main__':
    t0()
