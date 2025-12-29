# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/19 11:38
Create User : 19410
Desc : xxx
"""
import torch
from torchvision import models


def t0():
    net = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    img = torch.randn(1, 3, 224, 224)
    r = net(img)
    print(r.shape)


def t1():
    # net = models.detection.fasterrcnn_resnet50_fpn(
    #     weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    # )
    net = models.detection.ssd300_vgg16(
        weights=models.detection.SSD300_VGG16_Weights.DEFAULT
    )

    print(net)


if __name__ == '__main__':
    t1()
