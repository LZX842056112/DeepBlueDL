# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/22 10:05
Create User : 19410
Desc : YOLOv12基于 ultralytics 的使用
"""
import os
from typing import List

import cv2 as cv
from ultralytics.utils.metrics import DetMetrics

from ultralytics.engine.results import Results

from ultralytics import YOLO


# noinspection DuplicatedCode
def training():
    yaml_path = "./configs/yolo12s.yaml"
    model = YOLO(yaml_path)
    model.load("./configs/yolo12s.pt")  # 模型参数恢复(迁移的参数恢复)

    # https://docs.ultralytics.com/zh/usage/cfg/#train-settings
    data = r"datasets/coco8.yaml"
    results = model.train(
        data=data,
        epochs=3,
        imgsz=640, batch=2, device="cpu",
        workers=0, cos_lr=True, mixup=0.3,
        project="./runs/yolov12/detect",
        name=f"{os.path.splitext(os.path.basename(data))[0]}_train",
    )
    print(results)


# noinspection DuplicatedCode
def predicting():
    model_file = r".\configs\yolo12s.pt"

    # 1. 加载模型/恢复模型
    model = YOLO(model=model_file, task="detect", verbose=True)
    if model_file.endswith(".pt"):
        model.fuse()
        model.info(detailed=False, verbose=True)

    print("=" * 100)
    # 2. 预测
    results: List[Results] = model("./images/bus.jpg")
    print(len(results))
    print(type(results[0]))
    result = results[0]
    print(result.boxes.xywhn)  # 边框坐标预测值 百分比的值
    print(result.boxes.cls)  # 预测类别
    print(result.boxes.conf)  # 预测概率
    print("=" * 100)
    print(result.boxes)

    result.show()  # 可视化
    img = result.plot()  # 转换成numpy的图像类型

    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# noinspection DuplicatedCode
def val():
    model_file = r".\configs\yolo12s.pt"

    # 恢复模型
    model = YOLO(model_file)
    model.fuse()  # 算子的融合(eg: BN+Conv、RepVGG....)

    # 评估
    # https://docs.ultralytics.com/zh/usage/cfg/#validation-settings
    data = r"datasets/coco8.yaml"
    metrics: DetMetrics = model.val(
        data=data,
        save_txt=True, save_conf=True,
        save_json=True, save_hybrid=True,
        conf=0.3, iou=0.4, device='cpu',
        project="./runs/yolov12/detect",
        name=f"{os.path.splitext(os.path.basename(data))[0]}_val"
    )
    print(type(metrics))
    print(metrics.box.map)
    print(metrics.box.map50)
    print(metrics.box.map75)
    print(metrics.box.maps)
    print(metrics.box.f1)


def export():
    model_file = r"./configs/yolo12s.pt"

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
    training()
    predicting()
    val()
    export()
