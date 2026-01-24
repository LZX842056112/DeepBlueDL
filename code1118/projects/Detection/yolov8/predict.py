# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/16 16:18
Create User : 19410
Desc : 模型预测
"""

import warnings
from typing import List

warnings.filterwarnings('ignore')

from PIL import Image
import numpy as np
import cv2 as cv
from ultralytics.engine.results import Results
from ultralytics import YOLO


def detect_run():
    model_file = r".\runs\python\detect\mnist_train\weights\best.pt"
    model_file = r".\runs\python\detect\mnist_train\weights\best.onnx"

    # 1. 加载模型/恢复模型
    model = YOLO(model=model_file, task="detect", verbose=True)
    if model_file.endswith(".pt"):
        model.fuse()
        model.info(detailed=False, verbose=True)

    print("=" * 100)
    # 2. 预测
    results: List[Results] = model("./images/420741_u.png")
    # results: List[Results] = model("./images")
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


def cls_run():
    # 1. 加载模型
    model_file = r"./configs/yolov8n-cls.pt"
    model_file = r"./runs/python/classify/17flowers_train/weights/best.pt"
    model_file = r"./runs/python/classify/17flowers_train/weights/best.onnx"
    model = YOLO(model_file, verbose=True)
    if model_file.endswith(".pt"):
        model.fuse()  # 算子融合

    # 2. 对图像进行预测
    img_file = "./images/image_0180.jpg"
    results: List[Results] = model(img_file, imgsz=320)
    result = results[0]
    print("=" * 10)
    print(result.probs) # 预测概率
    print("=" * 10)
    print(result.names) # 映射类别名称
    print("=" * 10)
    print(result.probs.top1) # top1预测类别id
    print(result.probs.top1conf) # top1的预测概率
    print(result.names[result.probs.top1]) # top1预测类别名称


def segment_predict():
    model_file = r".\runs\python\segment\coco8-seg\weights\best.onnx"
    # model_file = r".\runs\python\segment\coco8-seg\weights\best.pt"
    model_file = r".\configs\yolov8n-seg.pt"
    # 1. 加载模型/恢复模型
    model = YOLO(model=model_file, task="segment", verbose=True)
    if model_file.endswith(".pt"):
        model.fuse()
        model.info(detailed=False, verbose=True)

    print("=" * 100)
    # 2. 预测
    results: List[Results] = model("./images/bus.jpg")
    print(type(results[0]))

    result = results[0]
    result.show()
    print("预测的物体边框信息:", "=" * 50)
    print(result.boxes.cls)  # 边框预测类别id
    print(result.boxes.conf)  # 边框预测类别置信度
    print(result.boxes.xyxyn)  # 边框预测坐标信息 百分比

    print("预测的物体mask/图像分割的信息:", "=" * 50)
    print(result.masks.xyn)  # 外边缘的百分比位置坐标

    # 可视化
    img = result.orig_img.copy()
    h, w = result.orig_shape
    cls_2_color = {}
    for i in range(len(result.boxes.cls)):  # 遍历每个预测边框
        cls = int(result.boxes.cls[i].item())

        # 绘制目标检测边框
        points = result.boxes.xyxyn[i].detach().numpy() * (w, h, w, h)  # 实际坐标点
        # x1y1 x1y2 x2y2 x2y1
        points = [points[0], points[1], points[0], points[3], points[2], points[3], points[2], points[1]]
        points = np.asarray(points).astype(np.int32)
        points = points.reshape((-1, 1, 2))
        try:
            color = cls_2_color[cls + 80]
        except KeyError:
            color = tuple(list(map(int, np.random.randint(0, 255, size=(3,)))))
            cls_2_color[cls + 80] = color
        cv.polylines(img, [points], isClosed=True, color=color, thickness=4)  # 绘制点

        # 绘制图像分割的边缘
        points = result.masks.xyn[i] * (w, h)  # 实际坐标点
        points = points.astype(np.int32)
        points = points.reshape((-1, 1, 2))
        try:
            color = cls_2_color[cls]
        except KeyError:
            color = tuple(list(map(int, np.random.randint(0, 255, size=(3,)))))
            cls_2_color[cls] = color
        cv.polylines(img, [points], isClosed=True, color=color, thickness=2)  # 绘制点

    cv.imshow('img', img)
    cv.imshow('box1', result.masks.data.detach().numpy()[0])
    cv.imshow('box2', result.masks.data.detach().numpy()[1])
    cv.waitKey(0)
    cv.destroyAllWindows()

    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image
        im.save('segment_result.jpg')  # save image


if __name__ == '__main__':
    # detect_run()
    # cls_run()
    segment_predict()
