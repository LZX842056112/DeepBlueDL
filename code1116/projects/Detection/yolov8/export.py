# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/16 16:33
Create User : 19410
Desc : 模型静态化转换输出
"""

import warnings

warnings.filterwarnings('ignore')

from ultralytics import YOLO, settings


def detect_export():
    """
    目标检测模型的静态转换
    :return:
    """
    model_file = r".\runs\python\detect\mnist_train\weights\best.pt"

    # 恢复模型
    model = YOLO(model_file)

    # 模型转换
    # https://docs.ultralytics.com/zh/usage/cfg/#export-settings
    # https://docs.ultralytics.com/zh/modes/export/#usage-examples
    model.export(format='onnx', imgsz=640, nms=False, device='cpu', opset=12, dynamic=False, max_det=10, conf=0.5)
    # model.export(format='onnx', imgsz=320, nms=False, device='cpu', opset=12, dynamic=True, max_det=10, conf=0.5)
    # model.export(format='onnx', imgsz=320, nms=True, device='cpu', dynamic=True, max_det=10, conf=0.5)
    # model.export(format='ncnn', imgsz=160, device='cpu')


if __name__ == '__main__':
    detect_export()
