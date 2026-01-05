# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/14 21:23
Create User : 19410
Desc : 构造Dataset和DataLoader相关的代码
"""
import os
from typing import List, Optional

import cv2 as cv
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


def load_datas(dir_path, class_names: Optional[List[str]] = None):
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

    image_paths, image_labels = [], []
    for cls_name in class_names:
        cls_path = os.path.join(dir_path, cls_name)
        if not os.path.exists(cls_path):
            continue

        # 遍历当前类别文件夹下的所有图像路径
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            image_paths.append(img_path)
            image_labels.append(class_name2id[cls_name])

    return image_paths, image_labels, class_names


# noinspection DuplicatedCode
def load_image(img_file, new_size):
    # 加载图像，将图像路径转换为图像对象
    img = cv.imread(img_file)
    # OpenCV原始BGR转RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # 图像大小缩放
    img = cv.resize(img, new_size)

    return img


class MyImageDataset(Dataset):
    def __init__(self, image_paths: List[str], image_labels: List[int], target_size):
        super().__init__()
        assert len(image_labels) == len(image_paths)
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.target_size = target_size

    def __getitem__(self, item):
        """
        获取item索引对应的样本x和y
        :param item: 索引id
        :return:
        """
        # 获取item这个索引对应的图像路径和图像标签id
        img_path: str = self.image_paths[item]
        img_label: int = self.image_labels[item]

        # 图像加载
        img = load_image(img_path, self.target_size)

        # 转换数据并输出
        # [H,W,C] --> [C,H,W]
        img = np.transpose(img, axes=(2, 0, 1))
        img = img / 255.0  # [0,255] -> [0,1]
        return {
            "image": torch.tensor(img, dtype=torch.float32),  # [C,H,W]形状的tensor对象
            "label": torch.tensor(img_label, dtype=torch.int64)  # 一个标量
        }

    def __len__(self):
        """
        获取当前数据集的总样本数目
        :return:
        """
        return len(self.image_labels)


def build_dataloader(data_dir, batch_size, shuffle, target_size=(100, 100), class_names=None):
    image_paths, image_labels, class_names = load_datas(data_dir, class_names)
    ds = MyImageDataset(
        image_paths=image_paths,
        image_labels=image_labels,
        target_size=target_size
    )
    dataloader = DataLoader(
        dataset=ds,
        batch_size=batch_size,  # 批次大小，也就是将几个样本合并成一个批次
        shuffle=shuffle,  # 是否打乱顺序
    )
    return ds, dataloader, class_names
