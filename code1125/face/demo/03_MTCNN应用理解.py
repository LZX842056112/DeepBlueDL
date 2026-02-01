# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/23 11:12
Create User : 19410
Desc : MTCNN三阶段的人脸检测
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from facenet_pytorch import MTCNN
from facenet_pytorch.models.utils import training


def t0():
    """
    MTCNN的基本认知
    :return:
    """
    device = torch.device("cpu")
    mtcnn = MTCNN(
        image_size=160,  # 输出人脸图像的区域大小，eg: 最终输出image_size*image_size
        margin=16,  # 在截取人脸区域的时候，往外扩展多少个像素
        min_face_size=20,  # 最小的人脸大小(在当前原始图像中最小人脸大小)
        thresholds=[0.6, 0.7, 0.7],  # P-、R-、O-Net三个网络后处理时候的阈值（分类）
        factor=0.709,  # 图像金字塔的缩放比例
        post_process=True,  # 最终返回的face tensor对象是否过标准化处理((x-127.5)/128)
        select_largest=True,  # 当keep_all为False，并且selection_method为None的时候，该参数生效；True表示获取区域中范围最大的人脸；False表示选择概率最大的人脸返回
        selection_method=None,  # 当keep_all为False的时候生效，直接给定选择的处理字符串
        keep_all=True,  # 是否保留所有检测图像，False的时候表示一张图像仅返回预测概率最大区域(只返回一个人脸图像)
        # keep_all=False,  # 是否保留所有检测图像，False的时候表示一张图像仅返回预测概率最大区域(只返回一个人脸图像)
        device=device
    )

    img_path = "./datas/images/faces2.png"
    save_path = f'./output/mtcnn/{os.path.basename(img_path).replace(".", "_")}'
    os.makedirs(save_path, exist_ok=True)
    img = Image.open(img_path)
    # faces: 人脸区域对应的tensor对象, [?, 3, image_size, image_size]
    # batch_probs: 返回区域属于人脸的概率值 [?]
    faces, batch_probs = mtcnn.forward(
        img,
        save_path=os.path.join(save_path, os.path.basename(img_path)),  # 给定截取人脸区别，并保存到磁盘
        return_prob=True
    )
    print(type(faces))
    print(faces.shape)

def t1():
    """
    基于现有的人脸检测模型生成人脸检测模型训练的数据
    :return:
    """
    device = torch.device("cpu")
    mtcnn = MTCNN(
        image_size=80,  # 输出人脸图像的区域大小，eg: 最终输出160*160
        margin=8,  # 在截取人脸区域的时候，往外扩展多少个像素
        min_face_size=20,  # 最小的人脸大小
        thresholds=[0.6, 0.7, 0.7],  # P-、R-、O-Net三个网络后处理时候的阈值
        factor=0.709,  # 图像金字塔的缩放比例
        post_process=True,
        select_largest=True,  # 当keep_all为False，并且selection_method为None的时候，该参数生效；True表示获取区域中范围最大的人脸；False表示选择概率最大的人脸返回
        selection_method=None,  # 当keep_all为False的时候生效，直接给定选择的处理字符串
        keep_all=True,  # 是否保留所有检测图像
        device=device
    )

    # === 下列代码主要功能是: 构造训练模型的数据标签
    input_dir = r"./datas/WIDER_FACE/images"
    label_dir = r"./datas/WIDER_FACE/labels"
    os.makedirs(label_dir, exist_ok=True)

    names = os.listdir(input_dir)
    for name in names:
        img_path = os.path.join(input_dir, name)
        img = Image.open(img_path)
        w, h = img.size
        # 仅做目标检测
        batch_boxes, batch_probs, batch_points = mtcnn.detect(img, landmarks=True)
        if batch_boxes is None:
            continue

        # 针对检测结果进行处理
        label_path = os.path.join(label_dir, f"{os.path.splitext(name)[0]}.txt")
        with open(label_path, 'w', encoding='utf-8') as writer:
            n = len(batch_boxes)
            for i in range(n):
                box = batch_boxes[i]  # 检测得到的边框 x1 y1 x2 y2
                box = box / [w, h, w, h]
                box = box.clip(0.0, 1.0)
                x1, y1, x2, y2 = box
                if x2 <= x1 or y2 <= y1:
                    continue
                points = batch_points[i]
                points = points / [[w, h]]
                points = points.clip(0.0, 1.0)
                if np.all(points > box[:2]) and np.all(np.all(points < box[2:])):
                    box[2] = x2 - x1  # box w
                    box[3] = y2 - y1  # box h
                    box[0] = box[0] + box[2] / 2  # box center x
                    box[1] = box[1] + box[3] / 2  # box center y

                    box = ' '.join(map(lambda t: f"{t:.4f}", box))
                    points = ' '.join(map(lambda t: f"{t:.4f}", points.reshape(-1)))
                    writer.writelines(f"0 {box} {points}\n")


def t2():
    """
    正常用于后续人脸特征向量提取的数据
    :return:
    """
    # 给定原始人脸数据所在文件夹，要求：文件夹内每个子文件是一个用户，每个子文件中就是该用户的图像数据
    data_dir = './datas/LFW/images'

    batch_size = 32
    epochs = 8
    workers = 0 if os.name == 'nt' else 8

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709,
        post_process=True, keep_all=False, device=device
    )

    dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
    dataset.samples = [
        (p, p.replace(data_dir, data_dir + '_cropped'))
        for p, _ in dataset.samples
    ]

    loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        collate_fn=training.collate_pil
    )

    # 生成人脸区域图像
    for i, (x, y) in enumerate(loader):
        mtcnn(x, save_path=y)
        print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')


if __name__ == '__main__':
    t0()
