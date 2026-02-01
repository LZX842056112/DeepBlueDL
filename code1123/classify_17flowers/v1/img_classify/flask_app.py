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

from PIL import Image
from flask import Flask, request, jsonify
from .predictor import Predictor

# 1. 定义app应用对象
app = Flask(__name__)
p = Predictor(
    model_file=os.environ['MODEL_FILE']
)


def base64_image_str_2_image(img_str):
    img_content = base64.b64decode(img_str)  # 解码还原
    image = Image.open(BytesIO(img_content))
    image = image.convert("RGB")
    return image

@app.route("/")
def index():
    return "提示信息：当前是一个模型预测后端API案例服务--图像分类"


# noinspection PyBroadException
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    try:
        # 1. 获取参数字典对象
        _args = {}
        if request.method == 'GET':
            # 当请求方式为GET的时候，参数直接从request.args(args是一个字典)中按照参数名称直接获取即可
            _args.update(request.args)
        elif request.method == 'POST':
            # 当请求方式为POST的时候，一般的参数直接从request.form(form是一个字典)中按照参数名称直接获取即可
            _args.update(request.form)
            try:
                # 当请求方式为Post，并且参数格式为json的时候，那么flask需要从request.json中获取(也是一个字典)
                _args.update(request.json)
            except:
                pass
        else:
            return jsonify({'code': 2, 'msg': '当前仅支持GET和POST请求方式'})

        # 2. 从字典中获取当前模型需要的入参
        b64_img_str = _args.get('image')
        if b64_img_str is None or len(b64_img_str) == 0:
            return jsonify({'code': 3, 'msg': f'请必须给定入参image，当前入参名称列表为: {_args.keys()}'})

        # 3. 调用编写好的模型推理方法
        result = p.predict(img_file=base64_image_str_2_image(b64_img_str))
        # print(result)

        # 4. 结构转换输出
        return jsonify({'code': 0, 'msg': '成功', 'data': result})
    except Exception as e:
        logging.error(f"服务器执行异常 {e}", exc_info=e)
        return jsonify({'code': 1, 'msg': f'服务器执行异常 {e}'})
