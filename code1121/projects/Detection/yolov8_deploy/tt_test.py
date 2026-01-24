# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/9 16:41
Create User : 19410
Desc : xxx
"""

import base64

import cv2 as cv
import requests

# DETECTION_URL = "http://127.0.0.1:9991/predict"
DETECTION_URL = "http://118.31.246.133:9991/predict"


def t0():
    from predictor import  YOLOV8Predictor, NewYOLOV8Predictor

    """
    [{'x1': 277, 'y1': 213, 'x2': 314, 'y2': 264, 'conf': 0.8489999771118164, 'id': 2}, {'x1': 33, 'y1': 36, 'x2': 45, 'y2': 52, 'conf': 0.671999990940094, 'id': 3}, {'x1': 48, 'y1': 206, 'x2': 74, 'y2': 234, 'conf': 0.6579999923706055, 'id': 4}, {'x1': 117, 'y1': 103, 'x2': 154, 'y2': 147, 'conf': 0.6309999823570251, 'id': 5}, {'x1': 42, 'y1': 115, 'x2': 63, 'y2': 140, 'conf': 0.546999990940094, 'id': 2}, {'x1': 150, 'y1': 50, 'x2': 173, 'y2': 112, 'conf': 0.4950000047683716, 'id': 1}, {'x1': 209, 'y1': 193, 'x2': 234, 'y2': 234, 'conf': 0.4390000104904175, 'id': 7}]
    """
    p = NewYOLOV8Predictor(
        model_file="./mnist_best.onnx",
        img_size=640
    )
    img_file = "./1223420_s.png"
    img = cv.cvtColor(cv.imread(img_file), cv.COLOR_BGR2RGB)
    r = p.predict(img=img)
    print(r)

def t1():
    img_file = "./1223420_s.png"

    # Read image
    with open(img_file, "rb") as f:
        image_data = f.read()  # 图像二进制数据
    img_base64 = base64.b64encode(image_data)  # 将二进制数据转换为字符串数据
    # print(img_base64)

    response = requests.post(
        DETECTION_URL,
        json={
            "image": str(img_base64, encoding='utf-8')
        }
    )

    if response.status_code == 200:
        print(response.json())
    else:
        print(f"调用服务器异常 {response.status_code}")


if __name__ == '__main__':
    t1()