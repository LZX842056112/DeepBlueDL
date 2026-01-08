# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/26 15:38
Create User : 19410
Desc : xxx
"""

from torchvision import datasets

ds = datasets.MNIST(
    root='datas/MNIST',  # 数据保存的文件夹路径
    download=True  # 如果在给定的文件夹中不存在数据，那么进行数据下载
)
t = ds[10]
print(type(t[0]))
print(type(t[1]))

print(f"类型:{t[1]}")
t[0].show(f"图像_{t[1]}")