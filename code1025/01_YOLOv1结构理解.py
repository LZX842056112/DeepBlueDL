# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/18 10:54
Create User : 19410
Desc : 深度学习思路2
"""

import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, num_classes=17, S=7, B=2):
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

        # 决策分支
        self.classify = nn.Sequential(
            nn.Linear(7 * 7 * 128, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, S * S * (num_classes + B * (1 + 4))),
        )
        self.C = num_classes
        self.S = S
        self.B = B

    def forward(self, image):
        """
        :param image: [N,C,H,W]
        :return: [N,17]
        """
        z = self.features(image)  # [N,3,H,W] -> [N,128,7,7]
        z = torch.flatten(z, 1, -1)  # [N,128,7,7] -> [N, 128*7*7]

        score = self.classify(z)  # [N, 128*7*7] -> [N, S * S * (num_classes + B * (1 + 4))]
        score = score.reshape(-1, self.S, self.S, self.C + self.B * (1 + 4))
        return score


if __name__ == '__main__':
    # NOTE: 假定一张图像中最多存在一个物体
    c = 20
    net = Network(num_classes=c, S=7, B=2)
    img = torch.rand(8, 3, 448, 448)  # 8个样本
    score = net(img)
    print(score.shape)  # 8个样本/图像，每个样本/图像种的物体属于各个类别的置信度

    pred_box1_prob, pred_box1_loc, pred_box2_prob, pred_box2_loc, pred_grad_prob =torch.split(score, [1,4,1,4,c], dim=-1)
    pred_box1_prob = pred_box1_prob.sigmoid()
    pred_box2_prob = pred_box2_prob.sigmoid()
    pred_grad_prob = pred_grad_prob.sigmoid()
    pred_box1_prob = pred_box1_prob * pred_grad_prob
    pred_box2_prob = pred_box2_prob * pred_grad_prob
    print(pred_box1_loc.shape)
