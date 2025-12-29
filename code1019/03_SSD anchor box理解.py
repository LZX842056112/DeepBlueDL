# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/19 14:32
Create User : 19410
Desc : xxx
"""

import os
import cv2 as cv
import torch

from PIL import Image

from torchvision import models, transforms

import numpy as np


def generator_box(fms, fmk, fmwh, s=(0, 300), cnt=4):
    x = (np.arange(0, fms) + 0.5) * fmk
    y = (np.arange(0, fms) + 0.5) * fmk
    x, y = np.meshgrid(x, y)
    cxy = np.stack((x, y), axis=-1)  # [h,w,2]

    wh = [
        [fmwh, fmwh],
        [fmwh / np.sqrt(2), fmwh * np.sqrt(2)],
        [fmwh * np.sqrt(2), fmwh / np.sqrt(2)],
        [fmwh * 2, fmwh * 2]
    ]
    if cnt == 6:
        wh.extend([
            [fmwh / np.sqrt(3), fmwh * np.sqrt(3)],
            [fmwh * np.sqrt(3), fmwh / np.sqrt(3)],
        ])
    wh = np.asarray(wh)
    wh = wh * fmk  # [n,2]

    cxy = cxy[:, :, None]  # [h,w,1,2]
    wh = wh[None, None]  # [1,1,k,2]

    x1y1 = cxy - wh / 2.0
    x2y2 = cxy + wh / 2.0

    x1y1x2y2 = np.concatenate([x1y1, x2y2], axis=-1)
    x1y1x2y2 = x1y1x2y2.clip(s[0], s[1])

    return x1y1x2y2.reshape((-1, 4))


def stage1():
    """
    负责将图像按照不同的滑动窗口产生不同的子图像
    :return:
    """
    img_path = "dog.jpeg"
    img = cv.imread(img_path)
    img = cv.resize(img, (300, 300))

    output_dir = "./output/ssd_anchor_box/" + os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    # 输出一个完整的图像
    cv.imwrite(os.path.join(output_dir, f'all.png'), img)

    h, w, c = img.shape

    boxs = []
    for fms, fmk, fmwh, cnt in [
        [38, 8, 3.0, 4],
        [19, 16, 3.0, 6],
        [10, 32, 3.0, 6],
        [5, 64, 3.0, 6],
        [3, 100, 2.3, 4],
        [1, 300, 0.9, 4]
    ]:
        boxs.append(generator_box(fms, fmk, fmwh, cnt=cnt))
    boxs = np.concatenate(boxs)
    boxs = boxs.astype('int')

    k = 0
    for x1, y1, x2, y2 in boxs:
        img_ = img[y1:y2, x1:x2, :]
        print(img_.shape)
        cv.imwrite(os.path.join(output_dir, f'{k:06d}_{x2 - x1}_{y2 - y1}_{x1}_{y1}.png'), img_)
        k += 1

    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    stage1()
