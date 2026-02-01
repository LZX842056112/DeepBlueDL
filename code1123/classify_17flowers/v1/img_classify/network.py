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

    def forward(self, x, return_features=False):
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

        if return_features:
            return score, x
        else:
            return score

class VGGImageClassifyNetwork(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        # 将训练好的vgg模型参数作为我们当前任务的初始参数  -- 模型参数的迁移
        self.vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        del self.vgg.classifier[-1] # 删除最后一个全连接操作

        # 针对迁移的模型这部分的参数来讲，模型参数的更新有以下几种方式：
        # 方式1：不区分迁移模块的参数和非迁移模块的参数，共同更新

        # 方式2：针对迁移模块的参数不进行参数更新 --> 模型参数冻结
        # 2.1 冻结所有迁移模块参数
        # for name, param in self.vgg.named_parameters():
        #     print(f"冻结参数:{name}")
        #     param.requires_grad = False
        # 2.2 冻结部分迁移模块参数(一般冻结前面的层)
        # pass

        # 方式3： 针对迁移部分的参数仅进行微调(eg：学习率调整成非迁移模块学习的1/100, 一般学习率至少1e-4以下)
        # pass

        self.classify = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, image, return_features=False):
        # 1. 提取vgg特征
        features = self.vgg(image)
        # 2. 决策得到置信度
        score = self.classify(features)

        if return_features:
            return score, features
        else:
            return score

def build_network(**kwargs):
    return VGGImageClassifyNetwork(
        num_classes=kwargs['num_classes'],
        in_channels=kwargs['in_channels']
    )
