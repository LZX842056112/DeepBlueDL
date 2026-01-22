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


if __name__ == '__main__':
    detect_val()
