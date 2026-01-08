# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/18 10:09
Create User : 19410
Desc : 深度学习思路1
"""

import os
import cv2 as cv
import torch

from PIL import Image

from torchvision import models, transforms


def stage1():
    """
    负责将图像按照不同的滑动窗口产生不同的子图像
    :return:
    """
    img_path = "../code1026/datas/dog.png"
    img = cv.imread(img_path)
    output_dir = "./output/roi/" + os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    h, w, c = img.shape
    # sizes = [(100, 100), (130, 70), (70, 130), (200, 200)]
    # sizes = [(200, 200), (300, 300)]
    sizes = [(224, 224), (300, 300)]
    k = 0
    step = 50  # 10
    for h_size, w_size in sizes:
        for i in range(0, h - h_size, step):
            for j in range(0, w - w_size, step):
                img_ = img[i:i + h_size, j:j + w_size, :]
                print(img_.shape)
                cv.imwrite(os.path.join(output_dir, f'{k:06d}_{h_size}_{w_size}_{i}_{j}.png'), img_)
                k += 1

    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def stage2():
    """
    阶段2： 针对候选区域进行图像分类
    :return:
    """
    net = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    net.eval()

    img0 = Image.open("../code1018/output/roi/dog/000000_224_224_0_0.png")
    # img1 = Image.open("./output/roi/dog/000087_300_300_50_250.png")
    img1 = Image.open("../code1018/output/roi/dog/000132_224_224_140_260.png")

    ts = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((227, 227))
    ])
    img0 = ts(img0)
    img1 = ts(img1)
    print(img0.shape)
    print(img1.shape)
    print("获取相同区域的数据输入", "=" * 100)
    print(img0[0, 5, 210:220])
    print(img1[0, 10:20, 200:210])

    r0 = net.features(img0[None])
    r1 = net.features(img1[None])
    print(r0.shape)
    print(r1.shape)
    print("获取相同区域的数据输出", "=" * 100)
    print(r0[0, 5])
    print(r1[0, 5])

    r0 = net(img0[None])  # 正常来讲，img0应该对应预测为小狗
    r1 = net(img1[None])  # 正常来讲，img1应该对应预测为背景
    print("预测对应类别", "=" * 100)
    print(torch.argmax(r0, dim=1))
    print(torch.argmax(r1, dim=1))


if __name__ == '__main__':
    stage2()
