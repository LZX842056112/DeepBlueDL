# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/10 21:28
Create User : 19410
Desc : flask 案例1
命令行启动: flask --app flask_demo01 run

(cpu_default) PS D:\workspaces\study\CV202502\code1010> flask --app flask_demo01 run
 * Serving Flask app 'flask_demo01'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit

"""

from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello, World!"


def print_infos():
    print(f"当前请求方法为: {request.method}")
    print(f"当前GET参数列表为:{request.args}")
    print(f"当前Post form参数列表为:{request.form}")
    try:
        print(f"当前Post json参数列表为:{request.json}")
    except Exception as e:
        print(f"当前Post json参数列表异常:{e}")
    print("=" * 100)


@app.route("/predict/get")
def get_predict():
    print_infos()
    return jsonify({'code': 0, 'msg': 'get invoke'})


@app.route("/predict/post", methods=['POST'])
def post_predict():
    print_infos()
    return jsonify({'code': 0, 'msg': 'post invoke'})
