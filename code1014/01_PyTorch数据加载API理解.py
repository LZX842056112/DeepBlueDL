# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/14 20:41
Create User : 19410
Desc : xxx
"""
from typing import List
import cv2 as cv
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader


class A:
    def __getitem__(self, item):
        print(f"入参为:{item}")
        return "nihao"


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


if __name__ == '__main__':
    # a = A()
    # r = a[10]
    # print(r)

    ds = MyImageDataset(
        image_paths=[
            r"../code0923/17flowers/val/c1/image_0001.jpg",
            r"../code0923/17flowers/val/c1/image_0002.jpg",
            r"../code0923/17flowers/val/c1/image_0003.jpg",
            r"../code0923/17flowers/val/c1/image_0004.jpg",
            r"../code0923/17flowers/val/c1/image_0005.jpg",
            r"../code0923/17flowers/val/c1/image_0006.jpg",
            r"../code0923/17flowers/val/c1/image_0007.jpg",
            r"../code0923/17flowers/val/c1/image_0008.jpg",
        ],
        image_labels=[1, 2, 3, 4, 5, 6, 7, 8],
        target_size=(100, 100)
    )
    print(len(ds))
    print(ds[1])
    print(ds[5])

    """
    def __init__(self, dataset: Dataset[T_co], batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = None, sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[List], Iterable[List], None] = None,
                 num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None, generator=None,
                 *, prefetch_factor: Optional[int] = None,
                 persistent_workers: bool = False,
                 pin_memory_device: str = ""):
    """
    dataloader = DataLoader(
        dataset=ds,
        batch_size=3,  # 批次大小，也就是将几个样本合并成一个批次
        shuffle=False,  # 是否打乱顺序
    )
    for batch in dataloader:
        print(batch)
        print(batch['image'].shape)
        print(batch['label'].shape)
