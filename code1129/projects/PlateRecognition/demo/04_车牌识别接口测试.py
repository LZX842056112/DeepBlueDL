# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/29 14:27
Create User : 19410
Desc : xxx
"""
import base64
import os

import requests
from PIL import Image

# 车牌号码识别的服务接口 --> 包括区域和号码的提取
URL = "http://118.31.246.133:9993/plate/predict"
IMG_DETECT_URL = "http://118.31.246.133:9992/predict"


def image_base64(image_path, print_data=True):
    with open(image_path, "rb") as reader:
        img_content = reader.read()  # 加载图像的所有二进制数据
        img_base64_content = base64.b64encode(img_content)  # 编码
        if print_data:
            print(img_base64_content)
        return img_base64_content


def invoke_remote_api(url, data):
    response = requests.post(
        url=url,
        data=data
    )
    if response.status_code == 200:
        result = response.json()
        print(result)
        print(type(result))
        return result
    else:
        raise ValueError("调用服务器异常")



def encode(img_path) -> str:
    with open(img_path, "rb") as reader:
        img_content = reader.read()  # 加载图像的所有二进制数据
        img_base64_content = base64.b64encode(img_content)  # 编码
        # print(img_base64_content)
        return str(img_base64_content, encoding='utf-8')


def extract_plate_and_save(img_path, img):
    base64_image_str = encode(img_path)

    r = requests.post(
        url=IMG_DETECT_URL,
        json={
            'image': base64_image_str
        }
    )
    if r.status_code == 200:
        data = r.json()['data']
        if len(data) == 1:
            data = data[0]
            x1, x2, y1, y2 = data['x1'], data['x2'], data['y1'], data['y2']

            x1 = int(x1) - 10
            x2 = int(x2) + 10
            y1 = int(y1) - 2
            y2 = int(y2) + 2
            return img.crop((x1, y1, x2, y2))
    else:
        print(f"请求服务器网络异常:{r.status_code}")
    return None



def t0():
    image_path = "../datasets/plate_images/0.jpg"
    invoke_remote_api(
        URL,
        data={
            'image': image_base64(image_path, print_data=False)
        }
    )


def t1():
    _dir = "../datasets/plate_images"
    for _name in os.listdir(_dir):
        print("=" * 50)
        _file = os.path.join(_dir, _name)
        print(f"处理图像: {_name}")
        invoke_remote_api(
            URL,
            data={
                'image': image_base64(_file, print_data=False)
            }
        )


def t2():
    _dir = "../datasets/plate_images"
    for _name in os.listdir(_dir):
        print("=" * 50)
        print(f"处理图像: {_name}")
        _file = os.path.join(_dir, _name)
        _img = Image.open(_file)
        _plate_img = extract_plate_and_save(_file, _img)
        if _plate_img is not None:
            _plate_img.save(f"../datasets/plate_images_result/{_name}")

if __name__ == '__main__':
    t1()
