# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/9/23 20:35
Create User : 19410
Desc :
0923:
    1.
"""
import os
from typing import List, Optional

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageClassifyNetwork(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        """
        模型结构的初始化：主要负责定义模型结构中涉及到的模块对象
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 1/2
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 1/4
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 1/8
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.pool4 = nn.AdaptiveMaxPool2d(8)  # h=w=8

        self.classify = nn.Linear(in_features=64 * 8 * 8, out_features=num_classes)

    def forward(self, x):
        """
        前向执行方法
        :param x: [N,C,H,W] 批次图像数据，N个图像，每个图像C个通道，每个图像的大小为H*W; 其中C必须是固定的
        :return:
        """
        # 1. 卷积 + 激活 [N,C,H,W] --> [N,32,H,W]
        x = F.relu(self.conv1(x))
        # 2. 池化 [N,32,H,W] --> [N,32,H/2,W/2]
        x = self.pool1(x)
        # 3. 卷积 + 激活 [N,32,H/2,W/2] --> [N,64,H/2,W/2]
        x = F.relu(self.conv2(x))
        # 4. 池化 [N,64,H/2,W/2] --> [N,64,H/4,W/4]
        x = self.pool2(x)
        # 5. 卷积 + 激活 [N,64,H/4,W/4] --> [N,64,H/4,W/4]
        x = F.relu(self.conv3(x))
        # 6. 池化 [N,64,H/4,W/4] --> [N,64,H/8,W/8]
        x = self.pool3(x)
        # 7. 卷积 + 激活 [N,64,H/8,W/8] --> [N,64,H/8,W/8]
        x = F.relu(self.conv4(x))
        # 8. 池化 [N,64,H/8,W/8] --> [N,64,8,8]
        x = self.pool4(x)

        # 9. 扁平化 [N,64,8,8] --> [N,64*8*8]
        x = x.flatten(1, -1)

        # 10. 全连接 决策判断 得到每个样本属于各个类别的置信度 [N,num_classes]
        score = self.classify(x)
        return score


def load_images(dir_path, new_size=(100, 100), class_names: Optional[List[str]] = None):
    """
    进行图像数据加载，数据存储层级:
        dir_path
            dir_path/c0
                dir_path/c0/img0.png
                dir_path/c0/img1.jpg
                ....
            dir_path/c1
                dir_path/c1/img0.jpg
                dir_path/c1/img1.png
                ....
            dir_path/c2
                ....
            .....
    :return:
    """
    # 1. 获取当前文件夹下的所有类别以及各个类别对应的图像路径
    if class_names is None:
        class_names = os.listdir(dir_path)
    class_name2id = {cls_name: cls_id for cls_id, cls_name in enumerate(class_names)}
    # print(f"当前类别映射mapping为:{class_name2id}")
    images_path = []
    for cls_name in class_names:
        cls_path = os.path.join(dir_path, cls_name)
        if not os.path.exists(cls_path):
            continue

        # 遍历当前类别文件夹下的所有图像路径
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            images_path.append((img_path, cls_name))

    # 2. 遍历每个图像，进行图像加载，形成numpy格式
    images = []
    for img_path, cls_name in images_path:
        # 加载图像，将图像路径转换为图像对象
        img = cv.imread(img_path)
        # OpenCV原始BGR转RGB
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        images.append((img, cls_name))

    # 3. 将图像转换为tensor对象 --> 图像的处理
    tensors = []
    labels = []
    for img, cls_name in images:
        img = cv.resize(img, new_size)
        # [H,W,C] --> [C,H,W]
        img = np.transpose(img, axes=(2, 0, 1))
        # 转换为tensor对象
        tensor = torch.tensor(img, dtype=torch.float32)
        tensors.append(tensor)

        # 类别id获取
        cls_id = class_name2id[cls_name]
        labels.append(torch.tensor(cls_id))

    return tensors, labels, class_names


def training():
    # 1. 数据加载
    # 2. 数据处理
    #     PS: 数据就变成了tensor格式
    train_path = r"17flowers\train"
    val_path = r"17flowers\val"
    train_images, train_labels, class_names = load_images(train_path)
    val_images, val_labels, _ = load_images(val_path, class_names=class_names)
    print(f"当前数据类别列表:{class_names}")

    #     3. 模型训练
    #         3.1 创建 --> 需要人为构造出网络结构、优化器、损失函数
    #             模型初始化 ----> 构造网络的执行图(构建图中的各个模块)
    #             Loss Function的构造
    #             优化器构造
    net = ImageClassifyNetwork(num_classes=len(class_names))
    print(net)
    #         3.2 训练 --> 需要人为进行数据的遍历以及前向反向过程的代码编写
    #             3.2.1 前向过程的执行 ---->
    #                 属于网络的执行图的构建(模型的执行顺序)
    #                 loss的获取
    #             3.2.2 反向过程的执行 ----> 不需要人为构造（框架会帮我们完成）
    #                 + 梯度计算 + 参数的更新 + 梯度重置为0
    #             NOTE: 训练是一个循环的过程，所以在训练过程中会有模型评估和模型持久化的操作
    #     4. 模型评估
    #         4.1 需要人为进行数据遍历、模型的推理预测、预测结果的评估
    #         PS：一般情况下，模型评估可以在训练过程中进行评估操作
    #     5. 模型持久化保存磁盘
    pass


def tt01():
    # 测试网络结构
    net = ImageClassifyNetwork(num_classes=17)
    print(net)

    images = torch.randn(4, 3, 80, 100)
    scores = net(images)
    print(scores.shape)


if __name__ == '__main__':
    # training()
    tt01()
