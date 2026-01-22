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

if __name__ == '__main__':
    detect_run()





