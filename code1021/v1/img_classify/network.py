# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/14 20:03
Create User : 19410
Desc : 定义网络相关的代码结构
"""

import torch.nn as nn
from torchvision import models


class ImageClassifyNetwork(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        """
        模型结构的初始化：主要负责定义模型结构中涉及到的模块对象
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(8)
        )

        self.classify = nn.Linear(in_features=64 * 8 * 8, out_features=num_classes)

    def forward(self, x):
        """
        前向执行方法
        :param x: [N,C,H,W] 批次图像数据，N个图像，每个图像C个通道，每个图像的大小为H*W; 其中C必须是固定的
        :return:
        """
        # 1. 卷积特征提取 [N,C,H,W] --> [N,64,8,8]
        x = self.features(x)

        # 2. 扁平化 [N,64,8,8] --> [N,64*8*8]
        x = x.flatten(1, -1)

        # 3. 全连接 决策判断 得到每个样本属于各个类别的置信度 [N,num_classes]
        score = self.classify(x)
        return score

class VGGImageClassifyNetwork(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        # 将训练好的vgg模型参数作为我们当前任务的初始参数  -- 模型参数的迁移
        self.vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        del self.vgg.classifier[-1] # 删除最后一个全连接操作
        self.classify = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, image):
        # 1. 提取vgg特征
        features = self.vgg(image)
        # 2. 决策得到置信度
        score = self.classify(features)
        return score

def build_network(**kwargs):
    return VGGImageClassifyNetwork(
        num_classes=kwargs['num_classes'],
        in_channels=kwargs['in_channels']
    )
