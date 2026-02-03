# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/10 21:51
Create User : 19410
Desc : 一个基于Flask框架的模型部署http接口demo
"""
import base64
import logging
import os
from io import BytesIO
from typing import Optional

if 'nt' in os.name:
    os.environ['XDG_CACHE_HOME'] = r"D:\huggingface"
    os.environ['CACHE_HOME'] = r'D:\huggingface'
    os.environ['MODELSCOPE_CACHE'] = r'D:\huggingface\modelscope\hub'

import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify

from inner.runner import PlateService

# 1. 定义app应用对象
app = Flask(__name__)
plate_service: Optional[PlateService] = None


def base64_image_str_2_image(img_str):
    img_content = base64.b64decode(img_str)  # 解码还原
    image = Image.open(BytesIO(img_content))
    image = image.convert("RGB")
    return image


@app.errorhandler(UnidentifiedImageError)
def internal_error(error):
    return jsonify({'code': 1, 'msg': '请给定有效的base64编码的图像参数，当前图像参数恢复异常.'}), 500


@app.route("/")
def index():
    return "Index首页: 车牌识别后端模型接口"


@app.route("/plate/predict", methods=['GET', 'POST'])
def plate_predict():
    try:
        if request.method == 'GET':
            return jsonify({
                'code': 0,
                'msg': '使用Post请求，并给定image参数',
                'data': {
                    'url': '/plate/predict',
                    'method': 'post',
                    'params': {
                        'image': 'base64编码的图像字符串'
                    }
                }
            })
        else:
            # 1. 获取参数
            image = request.values.get('image')
            if image is None:
                return jsonify({
                    'code': 1,
                    'msg': '必须给定有效的image参数.'
                })
            image = base64_image_str_2_image(img_str=image)

            # 2. 车牌号码提取
            result = plate_service.predict(image=image)

            # 3. 结果的返回
            return jsonify(result)
    except Exception as e:
        logging.error("服务器异常", exc_info=e)
        return jsonify({
            'code': 2,
            'msg': f'服务器异常:{e}'
        })


if __name__ == '__main__':
    plate_service = PlateService()
    app.run(host="0.0.0.0", port=9993)
