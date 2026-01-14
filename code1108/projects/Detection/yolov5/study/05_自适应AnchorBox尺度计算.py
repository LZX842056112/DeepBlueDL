# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/8 15:16
Create User : 19410
Desc : xxx
"""
import numpy as np
import yaml

from models.yolo import Model
from utils.autoanchor import check_anchors, kmean_anchors
from utils.dataloaders import create_dataloader
from utils.general import colorstr


def t0(dataset, net, hyp, imgsz):
    print(net.model[-1].anchors * np.asarray([8, 16, 32]).reshape(3, 1, 1))
    """
    1. 先计算yaml中配置的尺度在给定imgsz大小图像的真实边框的BPR(Gw/Aw 和 Gh/Ah 在 1.0/hyp ~ hyp 之间的占比)
    2. 如果bpr大于0.98，那么直接不进行自适应的更新；如果bpr小于0.98，会出发调用在imgsz大小图像的真实边框进行聚类操作，产生新的anchor尺度
    3. 将anchor新尺度计算新的BPR，如果发现new BPR比原始的BPR好，那么直接替换；
    """
    check_anchors(
        dataset,
        model=net,
        thr=hyp['anchor_t'],
        imgsz=imgsz
    )
    print(net.model[-1].anchors * np.asarray([8, 16, 32]).reshape(3, 1, 1))


def run(tt_func):
    hyp = r"../data/hyps/hyp.scratch-low.yaml"
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict

    train_loader, dataset = create_dataloader(
        r"D:/datas/coco128/images/train2017",  # 训练数据所在文件夹路径 --> 图像所在的文件夹
        320,  # 图像尺度大小 --> 最终输入到模型的图像大小
        batch_size=2,  # 最终迭代时候的样本批次大小
        stride=32,  # 模型的缩放比例
        single_cls=False,  # 是否是单类别的数据集，如果设置为True，会强制将所有box设定为类别0
        hyp=hyp,  # 超参数
        augment=True,  # 是否数据增强
        cache=None,  # 是否缓存数据
        rect=False,
        rank=-1,
        workers=0,  # 构造DataLoader对象的时候，给定是否并行加载数据
        image_weights=None,
        quad=False,
        prefix=colorstr('train: '),
        shuffle=True  # 是否打乱顺序
    )

    # cfg = "../models/yolov5s.yaml"
    cfg = "../models/yolov5s_copy_09.yaml"
    model = Model(
        cfg=cfg,  # 给定模型对应的结构配置信息，可以是dict字典或者yaml文件路径
        ch=3,  # 给定网络的输入通道数目，会被cfg中的参数ch覆盖
        nc=80,  # 类别的数量，会覆盖cfg中的参数nc的值
        anchors=None  # 给定先验框的尺度信息，如果非None，会覆盖cfg中的anchors参数
    )  # 基于配置文件进行对应的模型对象的创建

    tt_func(dataset, model, hyp, dataset.img_size)


if __name__ == '__main__':
    run(t0)
