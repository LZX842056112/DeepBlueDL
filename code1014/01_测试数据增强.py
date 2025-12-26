# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/16 21:17
Create User : 19410
Desc : xxx
"""

import random
import cv2 as cv

# noinspection DuplicatedCode
def load_image(img_file):
    # 加载图像，将图像路径转换为图像对象
    img = cv.imread(img_file)
    # # OpenCV原始BGR转RGB
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    return img


def image_augmentation(img):
    h, w, _ = img.shape  # 获取原始图像的高度和宽度
    cv.imshow('img', img)
    cv.waitKey(-1)
    cv.destroyAllWindows()

    if random.random() < 1.0:
        # 旋转 + 大小缩放
        angle = random.uniform(-20, 20)
        scale = random.uniform(0.7, 1.5)
        m = cv.getRotationMatrix2D(center=(h / 2, w / 2), angle=angle, scale=scale)
        img = cv.warpAffine(img, m, (w, h), borderValue=[0, 0, 0])
        cv.imshow('img', img)
        cv.waitKey(-1)
        cv.destroyAllWindows()

    if random.random() < 1.0:
        # 随机剪切
        new_h = min(h, int(h * random.uniform(0.7, 1.0)))
        new_w = min(w, int(w * random.uniform(0.7, 1.0)))
        start_h = max(0, int(0.5 * h - new_h / 2.0))
        end_h = min(h, start_h + new_h)
        start_w = max(0, int(0.5 * w - new_w / 2.0))
        end_w = min(w, start_w + new_w)
        img = img[start_h:end_h, start_w:end_w]
        cv.imshow('img', img)
        cv.waitKey(-1)
        cv.destroyAllWindows()

    if random.random() < 1.0:
        # 水平方向的翻转/交换
        img = cv.flip(img, 1)  # 左右翻转
        cv.imshow('img', img)
        cv.waitKey(-1)
        cv.destroyAllWindows()

    if random.random() < 1.0:
        # 垂直方向的翻转/交换
        img = cv.flip(img, 0)  # 上下翻转
        cv.imshow('img', img)
        cv.waitKey(-1)
        cv.destroyAllWindows()

    return img


if __name__ == '__main__':
    image_augmentation(
        load_image(
            img_file=r"../code0923/17flowers/val/c1/image_0001.jpg"
        )
    )