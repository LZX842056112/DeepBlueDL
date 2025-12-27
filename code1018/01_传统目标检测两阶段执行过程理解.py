# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/18 10:09
Create User : 19410
Desc : 传统思路
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
    img_path = "dog.jpeg"
    img = cv.imread(img_path)
    output_dir = "./output/roi/" + os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    h, w, c = img.shape
    # sizes = [(100, 100), (130, 70), (70, 130), (200, 200)]
    # sizes = [(200, 200), (300, 300)]
    sizes = [(224, 224), (300, 300)]
    k = 0
    step = 20  # 10
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
    roi_file = "./output/roi/dog/000000_224_224_0_0.png"

    # 1. 加载图像并且获取图像对应的特征向量
    img_features = None # load_img_features(roi_file) [1,n]

    # 2. 调用机器学习模型进行预测
    pred_idx = None # pred_idx = svm.predict(img_features) [1,n]
    pass

if __name__ == '__main__':
    stage1()
