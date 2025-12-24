# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/10 21:42
Create User : 19410
Desc : 模拟调用方
"""

import requests


def get_invoke(url, data):
    response = requests.get(url, params=data)
    if response.status_code == 200:
        print(f"Get请求返回结果: {url} --- {response.json()}")
    else:
        print(f"Get请求异常: {url} -- {response.status_code}")


def post_invoke(url, data, is_json=False):
    if is_json:
        # 参数以json格式进行传递，  python flask后端框架的参数解析必须从request.json获取
        response = requests.post(url, json=data)
    else:
        # 参数以form表格的方式进行传递，python flask后端框架的参数解析必须从request.form获取
        response = requests.post(url, data=data)

    if response.status_code == 200:
        print(f"Post请求返回结果: {url} --- {response.json()}")
    else:
        print(f"Post请求异常: {url} -- {response.status_code}")


def tt01():
    get_invoke(
        url="http://127.0.0.1:5000/predict/get",
        data={
            "name": "xh",
            "age": 23
        }
    )

    get_invoke(
        url="http://127.0.0.1:5000/predict/post",
        data={
            "name": "xh",
            "age": 23
        }
    )

    post_invoke(
        url="http://127.0.0.1:5000/predict/post",
        data={
            "address": "bj",
            "name": "xh",
            "age": 23
        },
        is_json=False
    )

    post_invoke(
        url="http://127.0.0.1:5000/predict/post",
        data={
            "address": "bj",
            "name": "xh",
            "age": 23,
            "id": 52
        },
        is_json=True
    )


def tt02():
    get_invoke(
        url="http://127.0.0.1:5000/predict",
        data={
            "image": r"D:\workspaces\study\CV202502\datas\c2_image_0120.jpg",
        }
    )

    post_invoke(
        url="http://127.0.0.1:5000/predict",
        data={
            "image": r"D:\workspaces\study\CV202502\datas\c2_image_0131.jpg",
        },
        # is_json=True
        is_json=False
    )


if __name__ == '__main__':
    tt02()
