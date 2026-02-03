# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/9 16:41
Create User : 19410
Desc : xxx
"""

import base64

import cv2 as cv
import requests

# DETECTION_URL = "http://127.0.0.1:9002/predict"
DETECTION_URL = "http://118.31.246.133:9992/predict"


def t0():
    from predictor import  YOLOv5Predictor

    p = YOLOv5Predictor(
        model_file="./best.onnx"
    )
    img_file = "../datasets/plate_images/10.png"
    img = cv.cvtColor(cv.imread(img_file), cv.COLOR_BGR2RGB)
    r = p.predict(img=img)
    print(r)

def t1():
    img_file = "../datasets/plate_images/10.png"

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