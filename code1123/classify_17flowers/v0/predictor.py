# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/10 20:13
Create User : 19410
Desc : 模型的预测器
"""
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


def load_img(img_file, new_size=(100, 100)):
    # 加载图像，将图像路径转换为图像对象
    img = cv.imread(img_file)
    # OpenCV原始BGR转RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # 图像大小缩放
    img = cv.resize(img, new_size)
    # [H,W,C] --> [C,H,W]
    img = np.transpose(img, axes=(2, 0, 1))
    img = img / 255.0  # [0,255] -> [0,1]
    # 转换为tensor对象
    tensor = torch.tensor(img, dtype=torch.float32)
    return tensor


class Predictor(object):
    def __init__(self, model_file='./output/model.pkl'):
        super().__init__()
        ckpt = torch.load(model_file, map_location='cpu')
        net_param = ckpt['net_param']  # 模型参数
        net = ImageClassifyNetwork(num_classes=17)
        # 当strict为True的时候，要求给定的参数列表和当前模型的参数列表必须完全匹配
        missing_keys, unexpected_keys = net.load_state_dict(net_param, strict=False)
        print(f"当前模型未进行参数恢复的相关参数名称列表:{missing_keys}")
        print(f"给定参数在当前模型中不存在:{unexpected_keys}")
        self.net = net

    @torch.no_grad()
    def predict(self, img_file: str, new_size=(100, 100)):
        # 2. 和训练采用相同的流程，对待预测的数据进行处理转换
        #     PS: 需要注意的是有一些数据增强的处理方法在推理的时候是不执行的；
        img = load_img(
            img_file=img_file,
            new_size=new_size
        )
        img = img[None]  # [C,H,W] --> [1,C,H,W]

        # 3. 调用模型的预测方法(前向过程)获取得到预测结果
        score = self.net(img)

        # 4. 后处理转换 --> 在模型预测结果的基础上额外的进行一些数据处理的工作
        pred_idx = torch.argmax(score, dim=-1)[0].item()

        # PS: 具体返回什么格式的数据，需要和需求方进行确认
        return int(pred_idx)


@torch.no_grad()
def tt01():
    # 1. 加载恢复模型(结构 + 参数，NOTE: 模型持久化的方式和模型恢复的方式必须是一一对应的)
    model_file = './output/model.pkl'
    ckpt = torch.load(model_file, map_location='cpu')
    net_param = ckpt['net_param']  # 模型参数
    # print(type(net_param))
    # print(net_param)
    net = ImageClassifyNetwork(num_classes=17)
    # 当strict为True的时候，要求给定的参数列表和当前模型的参数列表必须完全匹配
    missing_keys, unexpected_keys = net.load_state_dict(net_param, strict=False)
    print(f"当前模型未进行参数恢复的相关参数名称列表:{missing_keys}")
    print(f"给定参数在当前模型中不存在:{unexpected_keys}")

    # 2. 和训练采用相同的流程，对待预测的数据进行处理转换
    #     PS: 需要注意的是有一些数据增强的处理方法在推理的时候是不执行的；
    img = load_img(
        img_file="../../datas/c1_image_0016.jpg",
        new_size=(100, 100)
    )
    img = img[None]  # [C,H,W] --> [1,C,H,W]

    # 3. 调用模型的预测方法(前向过程)获取得到预测结果
    score = net(img)
    print(score.shape)

    # 4. 后处理转换 --> 在模型预测结果的基础上额外的进行一些数据处理的工作
    pred_idx = torch.argmax(score, dim=-1)[0].item()
    print(pred_idx)


def tt02():
    p = Predictor(
        model_file="./output/model.pkl"
    )

    r = p.predict(
        img_file="../../datas/c1_image_0016.jpg"
    )
    print(r)


def tt03():
    p = Predictor(
        model_file="./output/model.pkl"
    )
    while True:
        img_file = input("请输入图像路径:")
        if img_file == 'q':
            break
        r = p.predict(img_file)
        print(f"预测结果类别id:{r}")

if __name__ == '__main__':
    tt03()
