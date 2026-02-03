# -*- coding: utf-8 -*-

import base64
import os
import tempfile
from datetime import datetime

import requests
from PIL.Image import Image
import numpy as np


def encode(img_path) -> str:
    with open(img_path, "rb") as reader:
        img_content = reader.read()  # 加载图像的所有二进制数据
        img_base64_content = base64.b64encode(img_content)  # 编码
        # print(img_base64_content)
        return str(img_base64_content, encoding='utf-8')


def extract_plate_and_save(img_path, img):
    base64_image_str = encode(img_path)

    r = requests.post(
        url="http://118.31.246.133:9992/predict",
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
            return img[y1:y2, x1:x2, :]
    else:
        print(f"请求服务器网络异常:{r.status_code}")
    return None


class PlateDetectionModel(object):
    def __init__(self):
        super(PlateDetectionModel, self).__init__()
        self.tmp_save_dir = tempfile.TemporaryDirectory().name  # 临时文件夹
        os.makedirs(self.tmp_save_dir, exist_ok=True)

    def predict(self, image: Image):
        image = image.convert("RGB")
        # 1. 将image保存到临时的文件中
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = os.path.join(self.tmp_save_dir, f"plate_{now}.png")

        try:
            image.save(save_path)  # 保存

            # 2. 访问远程的服务，获取车牌区域
            plate_img = extract_plate_and_save(save_path, img=np.asarray(image))
            return plate_img
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)
