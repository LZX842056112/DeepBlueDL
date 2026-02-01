# -*- coding: utf-8 -*-
import base64
import logging
import os
import random
from datetime import datetime
from io import BytesIO

from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify

from .inner.runner import FaceRunner

app = Flask(__name__, static_folder="static")
project_root_dir = os.path.abspath(os.path.dirname(__file__))
face_runner = FaceRunner(
    vector_root_dir=os.path.join(app.static_folder, "embeddings"),
    project_root_dir=project_root_dir
)

face_img_save_dir_path = os.path.join(app.static_folder, "images")
os.makedirs(face_img_save_dir_path, exist_ok=True)


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
    return "Index首页: 简易的人脸对比&人脸检索后端服务"


@app.route("/face/compare", methods=['GET', 'POST'])
def face_compare():
    try:
        if request.method == 'GET':
            return jsonify({
                'code': 0,
                'msg': '人证对比请使用Post请求，并给定user_id和image两个参数',
                'data': {
                    'url': '/face/compare',
                    'method': 'post',
                    'params': {
                        'user_id': '用户id---用户唯一标识符',
                        'image': 'base64编码的图像字符串'
                    }
                }
            })
        else:
            # 1. 获取参数
            threshold = float(request.values.get('threshold', '0.8'))
            if not (0 < threshold < 1):
                threshold = 0.8
            how = request.values.get('how', 'any')
            user_id = request.values.get('user_id')
            image = request.values.get('image')
            if user_id is None or image is None:
                return jsonify({
                    'code': 1,
                    'msg': '必须给定有效的user_id和image参数.'
                })
            image = base64_image_str_2_image(img_str=image)

            # 2. 人证的对比
            result = face_runner.face_compare(
                user_id=user_id, image=image, threshold=threshold, how=how
            )

            # 3. 结果的返回
            return jsonify(result)
    except UnidentifiedImageError as e:
        raise e
    except Exception as e:
        logging.error("人证对比异常", exc_info=e)
        return jsonify({'code': 1, 'msg': f'人证对比服务器异常:{e}'})


@app.route("/face/search", methods=['GET', 'POST'])
def face_search():
    if request.method == 'GET':
        return jsonify({
            'code': 0,
            'msg': '人脸检索请使用Post请求，并给定image和k两个参数',
            'data': {
                'url': '/face/search',
                'method': 'post',
                'params': {
                    'image': 'base64编码的图像字符串',
                    'k': '获取最相似的K个匹配',
                    'threshold': '相似度阈值，默认为0.8'
                }
            }
        })
    else:
        # 1. 获取参数
        k = int(request.values.get('k', '1'))
        k = k if k > 0 else 1
        threshold = float(request.values.get('threshold', '0.8'))
        if not (0 < threshold < 1):
            threshold = 0.8
        image = request.values.get('image')
        if image is None:
            return jsonify({
                'code': 1,
                'msg': '必须给定有效的image参数.'
            })
        image = base64_image_str_2_image(img_str=image)

        # 2. 图像的检索
        result = face_runner.face_search(k=k, image=image, threshold=threshold)

        # 3. 结果的返回
        return jsonify(result)


@app.route("/face/register", methods=['GET', 'POST'])
def face_register():
    if request.method == 'GET':
        return jsonify({
            'code': 0,
            'msg': '人脸录入请使用Post请求，并给定image和user_id两个参数',
            'data': {
                'url': '/face/register',
                'method': 'post',
                'params': {
                    'user_id': '用户id---用户唯一标识符',
                    'image': 'base64编码的图像字符串'
                }
            }
        })
    else:
        # 1. 获取参数
        user_id = request.values.get('user_id')
        image = request.values.get('image')
        if user_id is None or image is None:
            return jsonify({
                'code': 1,
                'msg': '必须给定有效的user_id和image参数.'
            })
        image = base64_image_str_2_image(img_str=image)

        # 2. 图像的注册
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        rnd_num = random.randint(10000, 99999)
        save_path = os.path.join(face_img_save_dir_path, user_id, f"face_{now}_{rnd_num}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result = face_runner.face_register(user_id=user_id, image=image, save_path=save_path)

        # 3. 结果的返回
        return jsonify(result)
