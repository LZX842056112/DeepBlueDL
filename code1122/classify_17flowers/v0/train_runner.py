# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/9/23 20:35
Create User : 19410
Desc :
0923:
    1. 初步的数据加载逻辑：基于给定文件夹，加载对应的图像数据以及图像对应的类别id
    2. 构造了一个卷积神经网络:
        输入: [N,C,H,W] N个图像，每个图像大小为H*W，通道/"特征"/Feature Map 数目为C
        输出: [N, num_classes] N个图像，每个图像属于num_class个类别的置信度
0925:
    1. 交叉熵损失函数
    2. SGD优化器的创建
    3. 批次迭代进行参数的训练(前向+反向)
    4. 模型评估(准确率...)
    5. 模型持久化
"""
import os
from datetime import datetime
from typing import List, Optional

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn import metrics


class ImageClassifyNetwork(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        """
        模型结构的初始化：主要负责定义模型结构中涉及到的模块对象
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1, 1)
        # self.bn1 = nn.BatchNorm2d(32)
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
        # x = F.relu(self.bn1(self.conv1(x)))
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
        img = img / 255.0  # [0,255] -> [0,1]
        # 转换为tensor对象
        tensor = torch.tensor(img, dtype=torch.float32)
        tensors.append(tensor)

        # 类别id获取
        cls_id = class_name2id[cls_name]
        labels.append(torch.tensor(cls_id))

    return tensors, labels, class_names


# noinspection DuplicatedCode
def training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前运行设备:{device}")

    # 1. 数据加载
    # 2. 数据处理
    #     PS: 数据就变成了tensor格式
    train_path = r"D:\datas\17flowers\train"
    val_path = r"D:\datas\17flowers\val"
    # 训练数据加载
    train_images, train_labels, class_names = load_images(train_path)
    # 验证数据加载: PS --> 验证数据加载的时候，使用和训练数据完全一样的类别标签列表
    val_images, val_labels, _ = load_images(val_path, class_names=class_names)
    print(f"当前数据类别列表:{class_names}")

    #     3. 模型训练
    #         3.1 创建 --> 需要人为构造出网络结构、优化器、损失函数
    #             模型初始化 ----> 构造网络的执行图(构建图中的各个模块)
    #             Loss Function的构造
    #             优化器构造
    net = ImageClassifyNetwork(num_classes=len(class_names))
    net.to(device=device)
    print(net)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device=device)
    train_opt = optim.SGD(params=net.parameters(), lr=0.01, weight_decay=0)

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
    batch_size = 4
    val_batch_size = batch_size * 2
    total_epoch = 20
    train_numbers = len(train_images)  # 总训练数据数目
    val_numbers = len(val_images)  # 总评估数据数目

    for epoch in range(total_epoch):
        # 进入当前epoch的模型训练
        net.train()
        train_batch_numbers = train_numbers // batch_size  # 总的批次数目
        train_random_indexes = np.random.permutation(train_numbers)  # 随机的样本下标列表
        for batch_idx in range(train_batch_numbers):
            # a. 获取当前批次的数据
            batch_indexes = train_random_indexes[batch_size * batch_idx: batch_size * batch_idx + batch_size]
            batch_images = [train_images[_idx] for _idx in batch_indexes]
            batch_labels = [train_labels[_idx] for _idx in batch_indexes]
            batch_images = torch.stack(batch_images, dim=0)  # [N,C,H,W] 将N个图像list数据进行合并
            batch_labels = torch.stack(batch_labels, dim=0)  # [N,]
            batch_images = batch_images.to(device=device)
            batch_labels = batch_labels.to(device=device)

            # b. 前向过程执行
            scores = net(batch_images)  # 模型的前向执行，得到每个样本属于各个类别的预测置信度 [N,num_classes]
            loss = loss_fn(scores, batch_labels)

            # c. 反向过程执行
            loss.backward()  # 基于loss求解每个参数对应的梯度值
            train_opt.step()  # 当前train_opt中对应的所有可训练参数进行对应的更新(基于梯度值)
            train_opt.zero_grad()  # 将当前train_opt中包含的所有参数的对应梯度值重置为0

            print(f"Epoch {epoch} Batch {batch_idx}/{train_batch_numbers} Loss {loss.item():.3f}")



        # 模型评估
        net.eval()  # 标记当前模型进入评估执行逻辑
        with torch.no_grad():
            val_batch_numbers = val_numbers // val_batch_size  # 总的批次数目
            val_random_indexes = np.random.permutation(val_numbers)  # 随机的样本下标列表
            all_predicts, all_trues = [], []
            for val_batch_idx in range(val_batch_numbers):
                # a. 获取当前批次的数据
                si = val_batch_idx * val_batch_size
                ei = si + val_batch_size
                val_batch_indexes = val_random_indexes[si:ei]
                val_batch_images = [val_images[_idx] for _idx in val_batch_indexes]
                val_batch_labels = [val_labels[_idx] for _idx in val_batch_indexes]
                val_batch_images = torch.stack(val_batch_images, dim=0)  # [N,C,H,W] 将N个图像list数据进行合并
                val_batch_labels = torch.stack(val_batch_labels, dim=0)  # [N,]
                val_batch_images = val_batch_images.to(device=device)
                val_batch_labels = val_batch_labels.to(device=device)

                # b. 前向过程执行
                val_scores = net(val_batch_images)  # 模型的前向执行，得到每个样本属于各个类别的预测置信度 [N,num_classes]
                val_loss = loss_fn(val_scores, val_batch_labels)

                # 当前批次的预测标签id和实际标签id保存
                all_predicts.append(torch.argmax(val_scores.cpu(), dim=1).numpy())
                all_trues.append(val_batch_labels.cpu().numpy())

                print(f"Eval Epoch {epoch} Batch {val_batch_idx}/{val_batch_numbers} Loss {val_loss.item():.3f}")

            # 合并所有的
            all_predicts = np.concatenate(all_predicts, axis=0)
            all_trues = np.concatenate(all_trues, axis=0)

            val_acc = metrics.accuracy_score(all_trues, all_predicts)
            val_report = metrics.classification_report(all_trues, all_predicts)
            print("=" * 50)
            print(f"Eval Epoch {epoch} Accuracy {val_acc:.3f}")
            print(f"Eval Epoch {epoch} Class report:\n{val_report}\n")
            print("=" * 50)

        # # 针对最后的两个epoch，批次大小变大
        # if epoch >= total_epoch - 2:
        #     batch_size = 8

    # 5. 模型持久化保存磁盘
    obj = {
        'net_param': net.state_dict(),  # 将模型的参数以dict的形式保存下来
        'loss_param': loss_fn.state_dict(),
        'opt_param': train_opt.state_dict(),
        'date': datetime.now(),
        "class_names": class_names
    }
    model_file = './output/model.pkl'
    os.makedirs(os.path.abspath(os.path.dirname(model_file)), exist_ok=True)
    # torch.save底层是基于python的 pickle 库进行持久化操作；对应的加载API为 torch.load
    torch.save(obj, model_file)


def tt01():
    # 测试网络结构 + loss
    net = ImageClassifyNetwork(num_classes=17)
    loss_fn = nn.CrossEntropyLoss()
    print(net)

    images = torch.randn(4, 3, 80, 100)
    targets = torch.tensor([
        5, 1, 3, 6
    ])
    scores = net(images)  # 模型的前向执行，得到每个样本属于各个类别的预测置信度 [N,num_classes]
    print(scores.shape)
    loss = loss_fn(scores, targets)
    print(loss)


if __name__ == '__main__':
    training()
    # tt01()
