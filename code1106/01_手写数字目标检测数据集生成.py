# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/6 20:15
Create User : 19410
Desc : xxx
"""
import os
import random
import numpy as np

from torchvision import datasets
from PIL import Image


def transform_fun(_img: Image.Image):
    # 转换为灰度图像
    _img = _img.convert("L")
    # c. 随机选择的图像进行随机缩放(等比例缩放), 倍数：0.5~2
    if random.random() < 0.8:
        min_ratio, max_ratio = 0.5, 3.0
        size_ratio = np.clip(np.random.rand() * (max_ratio - min_ratio) + min_ratio, min_ratio, max_ratio)
        new_size = int(_img.size[0] * size_ratio)
        print(f"缩放:{new_size}")
        _img = _img.resize((new_size, new_size))
    # d. 随机选的图像进行随机的旋转操作，旋转角度: -30~30
    if random.random() < 0.5:
        angle = np.random.uniform(-30, 30)
        print(f"旋转:{angle}")
        _img = _img.rotate(angle=angle, fillcolor=0)
    return _img


dataset = datasets.MNIST(
    root='MNIST',  # 数据保存的文件夹路径
    train=True,
    # transform=
    download=True  # 如果在给定的文件夹中不存在数据，那么进行数据下载
)
n = len(dataset)  # 总的原始手写数字图像数目


def rnd_img():
    """
    随机产生一张图像
    :return: PIL.Image, int
    """
    idx = random.randint(0, n - 1)
    img, label = dataset[idx]
    return img, label


def paste_number_img(img_bg, k):
    img_bg_size = img_bg.shape  # (h,w)
    l, r, t = 30, 50, 80  # 允许粘贴图像的坐标范围以及间隔
    # l, r, t = 30, 50, 20
    x = list(range(l, img_bg_size[1] - r, t))
    y = list(range(l, img_bg_size[0] - r, t))
    x, y = np.meshgrid(x, y)
    points = np.stack([x.reshape(-1), y.reshape(-1)], axis=-1)
    rnd_points_offset = np.random.uniform(-10, 20, size=points.shape).round().astype(points.dtype)
    points = points + rnd_points_offset
    points = [tuple(map(int, v)) for v in points]
    labels = []

    for _ in range(k):
        # 随机产生一张图像
        img, label = rnd_img()
        new_img = transform_fun(img)
        w, h = new_img.size  # (w,h)
        print(f"图像大小为:{(w, h)}")

        # 粘贴这个图像
        x1, y1 = random.choice(points)  # 随机旋转一个坐上角点
        points.remove((x1, y1))  # 删除不能使用的点
        x2 = x1 + w
        y2 = y1 + h
        if y2 >= img_bg_size[0] or x2 >= img_bg_size[1]:
            continue

        new_img = np.clip(img_bg[y1:y2, x1:x2] + np.asarray(new_img), 0, 255)
        new_img = new_img.astype('uint8')
        img_bg[y1:y2, x1:x2] = new_img
        labels.append(str(label))

        if len(points) == 0:
            break

    return img_bg, ''.join(labels)


def paste_char_img(img_bg, k=1):
    import cv2 as cv

    h, w = img_bg.shape  # (h,w)
    chars = list('qazwsxedcrfvtgbyhnujmikolp')

    char_list = []
    for _ in range(k):
        char = random.choice(chars)

        cv.putText(
            img_bg,
            text=char,
            org=(np.random.randint(50, w - 50), np.random.randint(50, h - 50)),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=np.random.uniform(1.0, 3.0),
            color=255,
            thickness=2,
            lineType=cv.LINE_AA
        )
        char_list.append(char)

    return img_bg, ''.join(char_list)

#
# k = np.random.randint(5, 11)
#
# img_w = np.random.randint(100, 200) + k * 30
# img_h = np.random.randint(100, 200) + k * 30
# img_full = np.zeros((img_h, img_w), dtype='uint8')
# img_full, name = paste_number_img(img_full, k=k)
# img_full, char = paste_char_img(img_full, k=random.randint(1, 4))

# import cv2 as cv
#
# cv.imshow(f"{name}_{char}", img_full)
# cv.waitKey(-1)
# cv.destroyAllWindows()

output_dir = r"MNIST\1106\images" # 数据输出文件夹路径
os.makedirs(output_dir, exist_ok=True)

for _ in range(100):
    k = np.random.randint(5, 11)

    img_w = np.random.randint(100, 200) + k * 30
    img_h = np.random.randint(100, 200) + k * 30
    img_full = np.zeros((img_h, img_w), dtype='uint8')
    img_full, name = paste_number_img(img_full, k=k)
    img_full, char = paste_char_img(img_full, k=random.randint(1, 4))

    output_file = os.path.join(output_dir, f"{name}_{char}.png")
    Image.fromarray(img_full, "L").save(output_file)
