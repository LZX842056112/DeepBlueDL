# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/16 16:15
Create User : 19410
Desc : 模型评估
"""

import os
import warnings

warnings.filterwarnings('ignore')

from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics, SegmentMetrics, ClassifyMetrics, PoseMetrics, OBBMetrics


def detect_val():
    """
    目标检测的模型效果评估
    :return:
    """
    model_file = r".\runs\python\detect\mnist_train\weights\best.pt"

    # 恢复模型
    model = YOLO(model_file)
    model.fuse()  # 算子的融合(eg: BN+Conv、RepVGG....)

    # 评估
    # https://docs.ultralytics.com/zh/usage/cfg/#validation-settings
    data = r"./datasets/mnist.yaml"
    metrics: DetMetrics = model.val(
        data=data,
        save_txt=True, save_conf=True,
        save_json=True, save_hybrid=True,
        conf=0.3, iou=0.4, device='cpu',
        project="./runs/python/detect",
        name=f"{os.path.splitext(os.path.basename(data))[0]}_val"
    )
    print(type(metrics))
    print(metrics.box.map)
    print(metrics.box.map50)
    print(metrics.box.map75)
    print(metrics.box.maps)
    print(metrics.box.f1)


# noinspection PyTypeChecker
def cls_val():
    # 1. 加载模型
    model_file = r"./configs/yolov8n-cls.pt"
    model_file = r"./runs/python/classify/17flowers_train/weights/best.pt"
    model = YOLO(model_file, verbose=True)
    model.fuse()  # 算子融合

    # 2. 评估
    data = r"D:\datas\17flowers"
    metrics: ClassifyMetrics = model.val(
        data=data,
        save_txt=True, save_conf=True,
        save_json=True, save_hybrid=True,
        conf=0.3, iou=0.4, device='cpu',
        project="./runs/python/classify",
        name=f"{os.path.splitext(os.path.basename(data))[0]}_val"
    )
    print(metrics.top1)
    print(metrics.top5)


def segment_val():
    model_file = r".\runs\python\segment\coco8-seg\weights\best.pt"
    # 恢复模型
    model = YOLO(model_file, task="segment")
    model.fuse()  # 算子的融合(eg: BN+Conv、RepVGG....)

    # 评估
    # https://docs.ultralytics.com/zh/usage/cfg/#validation-settings
    data = r"./datasets/coco8-seg.yaml"
    # noinspection PyTypeChecker
    metrics: SegmentMetrics = model.val(
        data=data,
        save_txt=True, save_conf=True,
        save_json=True, save_hybrid=True,
        # conf=0.9, iou=0.7,
        device='cpu',
        project="./runs/python/segment",
        name=f"{os.path.splitext(os.path.basename(data))[0]}_val"
    )
    print(type(metrics))
    print("=" * 50)
    print(metrics.box.map)
    print(metrics.box.map50)
    print(metrics.box.map75)
    print(metrics.box.maps)
    print(metrics.box.f1)
    print("=" * 50)
    print(metrics.seg.map)
    print(metrics.seg.map50)
    print(metrics.seg.map75)
    print(metrics.seg.maps)
    print(metrics.seg.f1)


# noinspection PyTypeChecker
def keypoint_val():
    # 1. 加载恢复模型
    model = YOLO(model="./configs/yolov8n-pose.pt", task="pose", verbose=True)
    model.fuse()  # 算子融合

    # 评估
    metrics: PoseMetrics = model.val(
        data=r"./datasets/coco8-pose.yaml",
        save_json=True,
        # save_hybrid=True,
        # conf=0.9, iou=0.7,
        device='cpu'
    )
    print(type(metrics))

    print(metrics.box.map)
    print(metrics.box.map50)
    print(metrics.box.map75)
    print(metrics.box.maps)
    print(metrics.box.f1)

    print(metrics.pose.map)
    print(metrics.pose.map50)
    print(metrics.pose.map75)
    print(metrics.pose.maps)
    print(metrics.pose.f1)


# noinspection PyTypeChecker
def obb_val():
    model_file = "./configs/yolov8n-obb.pt"

    # 恢复模型
    model = YOLO(model_file)
    model.fuse()  # 算子的融合(eg: BN+Conv、RepVGG....)

    # 评估
    # https://docs.ultralytics.com/zh/usage/cfg/#validation-settings
    data = r"./datasets/dota8.yaml"
    metrics: OBBMetrics = model.val(
        data=data,
        save_txt=True, save_conf=True,
        save_json=True, save_hybrid=True,
        conf=0.9, iou=0.7, device='cpu',
        project="./runs/python/obb",
        name=f"{os.path.splitext(os.path.basename(data))[0]}_val"
    )
    print(type(metrics))
    print(metrics.box.map)
    print(metrics.box.map50)
    print(metrics.box.map75)
    print(metrics.box.maps)
    print(metrics.box.f1)


if __name__ == '__main__':
    # detect_val()
    # cls_val()
    # segment_val()
    # keypoint_val()
    obb_val()
