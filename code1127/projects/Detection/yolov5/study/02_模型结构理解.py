# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/2 10:29
Create User : 19410
Desc : xxx
"""
from pathlib import Path

import numpy as np
import cv2 as cv
import torch

from models.yolo import Model


@torch.no_grad()
def t0():
    ckpt_file = "./yolov5s.pt"
    ckpt = torch.load(ckpt_file, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
    model = Model(
        ckpt['model'].yaml,  # 模型结构信息 --> 如果当前没有给定，那么直接使用迁移模型的结构信息
        ch=3,  # 输入通道数
        nc=80,  # 类别数目
        anchors=None
    )  # create 基于给定的结构创建对应的模型
    model.eval()

    # 模型参数初始化
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    missing_keys, unexpected_keys = model.load_state_dict(csd, strict=False)  # load
    print(f"missing_keys : {missing_keys}")
    print(f"unexpected_keys : {unexpected_keys}")

    visualize_dir = Path('runs/visualize')
    visualize_dir.mkdir(parents=True, exist_ok=True)

    img = cv.imread(r'../data/images/bus.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (320, 320))
    img = np.transpose(img, (2, 0, 1))  # [h,w,c] -> [c,h,w]
    img = img[None]  # [c,h,w] -> [1,c,h,w]
    img = torch.tensor(img) / 255.0
    r = model(
        img,
        augment=False,  # 推理的时候是否做增强推理 --> 金字塔增强推理
        profile=False,  # 执行的时候是否计算每层的耗时信息
        visualize=False  # 可视化 查看每一层的输出特征图
        # visualize=visualize_dir  # 可视化 查看每一层的输出特征图
    )
    print(r[0].shape)


def t1():
    # 原始的YOLOv5s的模型结构配置文件
    # cfg = r"../models/yolov5s_copy_00.yaml"
    # 增加每一层的每个grid cell对应的anchor box边框数目 --> 不改变预测分支的数目 P3\P4\P5
    # cfg = r"../models/yolov5s_copy_01.yaml"
    # 减少每一层的每个grid cell对应的anchor box边框数目 --> 不改变预测分支的数目 P3\P4\P5
    # cfg = r"../models/yolov5s_copy_02.yaml"
    # 减少预测分支 --> 仅改决策头/更改主体的结构
    # cfg = r"../models/yolov5s_copy_03.yaml"
    # cfg = r"../models/yolov5s_copy_04.yaml"
    # cfg = r"../models/yolov5s_copy_05.yaml"
    # 增加预测分支 --> 假定P3层对应的feature map还是太深了，针对特小物体，采用P2层进行预测
    # cfg = r"../models/yolov5s_copy_06.yaml"

    # 使用新结构：实现方式1 --> 新结构可以基于现有的模块进行实现（类似YOLOv6的BiC结构）
    # cfg = r"../models/yolov5s_copy_07.yaml"

    # 使用新结构：可能需要新增模块
    """
    新增模块的步骤：--> 增加一个下采样模块 MP(池化 + 全连接)、更改C3模块
        1. 在common.py文件中新建模块，并测试模块
        2. 在yolo.py文件的模块解析函数parse_model中，增加当前新模块的解析逻辑
        3. 在当前的模型配置yaml文件中进行更改，使用新模块
        4. 测试
        5. 训练使用
            python train.py --data ./data/coco128_copy.yaml --weights ./study/yolov5s.pt  --batch-size 4 --workers 0 --epochs 3 --name m_coco128 --device cpu --cfg ./models/yolov5s_copy_08.yaml
    """
    cfg = r"../models/yolov5s_copy_08.yaml"

    model = Model(
        cfg,  # 模型结构信息 --> 如果当前没有给定，那么直接使用迁移模型的结构信息
        ch=3,  # 输入通道数
        nc=80,  # 类别数目
        anchors=None
    )
    print(model)
    # model.eval()

    # 训练时候的返回值
    # na: number of anchor --> 每个 锚点/grid cell 对应几个 anchor box/预测边框
    # nc: number of class --> 类别数目
    # N: 批次样本大小；H：feature map的高度；W：feature map的宽度；
    # 训练时候返回各个分支/各层对应的预测值, [N,na,H,W,nc+1+4]
    x = torch.rand(4, 3, 320, 320)
    r = model(x)
    print("=" * 100)
    if model.training:
        print(type(r))  # <class 'list'>
        print(len(r))  # 3
        for ri in r:
            print(ri.shape)
    else:
        print(type(r))  # <class 'tuple'>
        print(len(r))  # 2
        print(r[0].shape)  # torch.Size([4, 6300, 85]) 所有分支的预测结果合并[bs, sum(na*H*W), nc+1+4] 边框位置是转换后的
        for ri in r[1]:  # 和训练一样是三个分支的预测值
            print(ri.shape)

    model.model[-1].export = True
    torch.onnx.export(
        model.cpu().eval(),  # --dynamic only compatible with cpu
        x.cpu(),
        './model.onnx',
        verbose=False,
        opset_version=12,
        do_constant_folding=True,
        input_names=['x'],
        output_names=['y'],
        dynamic_axes=None
    )


if __name__ == '__main__':
    t1()
