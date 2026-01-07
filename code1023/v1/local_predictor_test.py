# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/23 21:03
Create User : 19410
Desc : xxx
"""
import base64

import requests


def image_base64(image_path, print_data=False):
    with open(image_path, "rb") as reader:
        img_content = reader.read()  # 加载图像的所有二进制数据
        img_base64_content = base64.b64encode(img_content)  # 编码
        if print_data:
            print(img_base64_content)
        return img_base64_content


def tt01():
    from img_classify.predictor import Predictor

    p = Predictor(
        model_file="./deploy/best.pkl"
    )
    r = p.predict(
        # r"../../datas/c1_image_0017.jpg"
        r"../../datas/c6_image_0417.jpg"
    )
    print(r)
    for k, v in r.items():
        print(k, "---", type(v))


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


def tt02():
    # url = "http://127.0.0.1:9002"
    url = "http://118.31.246.133:9002"

    path = r"D:\workspaces\study\CV202502\datas\c2_image_0131.jpg"
    path = r"D:\workspaces\study\CV202502\datas\c11_image_0817.jpg"

    post_invoke(
        url=f"{url}/predict",
        data={
            "image": image_base64(path),
        },
        # is_json=True
        is_json=False
    )


if __name__ == '__main__':
    tt02()
