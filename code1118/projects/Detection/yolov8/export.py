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


def classify_export():
    # 1. 加载模型/模型恢复
    model_file = r"./configs/yolov8n-cls.pt"
    model_file = r"./runs/python/classify/17flowers_train/weights/best.pt"
    model = YOLO(model_file, verbose=True)

    # 2. 导出
    model.export(format="onnx", dynamic=True, simplify=True)


def segment_export():
    model_file = r".\runs\python\segment\coco8-seg\weights\best.pt"

    # 恢复模型
    model = YOLO(model_file, task="segment")

    # 模型转换
    # https://docs.ultralytics.com/zh/usage/cfg/#export-settings
    # https://docs.ultralytics.com/zh/modes/export/#usage-examples
    # model.export(format='onnx', imgsz=320, nms=False, device='cpu', opset=12, dynamic=False, max_det=10, conf=0.5)
    # model.export(format='onnx', imgsz=320, nms=False, device='cpu', dynamic=True, max_det=10, conf=0.5)
    model.export(format='ncnn', imgsz=160, device='cpu')


if __name__ == '__main__':
    # detect_export()
    # classify_export()
    segment_export()
