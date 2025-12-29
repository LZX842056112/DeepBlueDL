# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/18 10:54
Create User : 19410
Desc : Faster RCNN
"""

import torch
import torch.nn as nn


class RPNNetwork(nn.Module):
    def __init__(self, in_channels, k=9):
        super().__init__()
        # 1. 3*3的卷积进行特征融合
        self.rpn_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU()
        )

        # 2. 两个分支
        self.classify_header = nn.Sequential(
            nn.Conv2d(256, 2 * k, stride=(1, 1), kernel_size=(1, 1), padding=0)
        )
        self.reg_header = nn.Sequential(
            nn.Conv2d(256, 4 * k, stride=(1, 1), kernel_size=(1, 1), padding=0)
        )

    def forward(self, x):
        """
        :param x: [N,C,H,W]
        :return:
        """
        x = self.rpn_3x3(x)  # [N,C,H,W] --> [N,256,H,W]
        n, _, h, w = x.shape

        cls = self.classify_header(x)  # [N,256,H,W] --> [N,18,H,W]
        cls = cls.reshape((n, -1, 2, h, w))
        cls = torch.permute(cls, (0, 3, 4, 1, 2))  # [N,H,W,k,2]
        cls = cls.reshape((n, -1, 2))  # [N, H*W*k, 2]

        reg = self.reg_header(x)  # [N,256,H,W] --> [N,36,H,W]
        reg = reg.reshape((n, -1, 4, h, w))
        reg = torch.permute(reg, (0, 3, 4, 1, 2))  # [N,H,W,k,4]
        reg = reg.reshape((n, -1, 4))  # [N, H*W*k, 4]

        return cls, reg


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # 特征提取模块 --> 负责针对输入图像，提取基础的通用的局部特征信息 ---> 一般采用VGG、ResNet、MobileNet....网络
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(7, 7))
        )

        # RPN网络部分
        self.rpn = RPNNetwork(128)

        # 决策分支
        self.classify_header = nn.Sequential(
            nn.Linear(7 * 7 * 128, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 17)  # 输入的是每个类别的置信度，17个类别
        )
        self.reg_header = nn.Sequential(
            nn.Linear(7 * 7 * 128, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 4)  # 输入的是每个边框坐标信息，x y w h
        )

    def forward(self, image):
        """
        :param image: [N,C,H,W]
        :return: [N,17]
        """
        z = self.features(image)  # [N,3,H,W] -> [N,128,7,7]

        # 执行RPN网络
        # rpn_cls [n,441,2] --> n个图像，每个图像有441个anchor box，每个box对应有物体和没有物体的预测置信度
        rpn_cls, rpn_reg = self.rpn(z)

        # TODO: 执行Fast RCNN逻辑 + 候选框过滤的一些逻辑（普通python代码，非模型代码）


        z = torch.flatten(z, 1, -1)  # [N,128,7,7] -> [N, 128*7*7]

        cs = self.classify_header(z)  # [N, 128*7*7] -> [N,17]
        rb = self.reg_header(z)
        return cs, rb


if __name__ == '__main__':
    # NOTE: 假定一张图像中最多存在一个物体
    net = Network()
    img = torch.rand(8, 3, 224, 224)  # 8个样本
    box_class_score, box_loc = net(img)
    print(box_class_score.shape)  # 8个样本/图像，每个样本/图像种的物体属于各个类别的置信度
    print(box_loc.shape)  # 8个样本/图像，每个样本/图像种物体对应的位置坐标信息xywh
