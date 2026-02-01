# -*- coding: utf-8 -*-

import base64

import requests

base_url = "http://127.0.0.1:9998"


def image_base64(image_path, print_data=False):
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
    print("\n", "=" * 50)
    print(f"调用 {url}")
    if response.status_code == 200:
        result = response.json()
        print(result)
        print(type(result))
    else:
        raise ValueError("调用服务器异常")


def register(user_id, img_path):
    invoke_remote_api(
        f'{base_url}/face/register',
        data={
            'user_id': user_id,
            'image': image_base64(img_path)
        }
    )


def face_compare(user_id, img_path, threshold=0.7):
    invoke_remote_api(
        f'{base_url}/face/compare',
        data={
            'user_id': user_id,
            'image': image_base64(img_path),
            'threshold': threshold
        }
    )


def face_search(img_path, k=1, threshold=0.6):
    invoke_remote_api(
        f'{base_url}/face/search',
        data={
            'image': image_base64(img_path),
            'k': k,
            'threshold': threshold
        }
    )


if __name__ == '__main__':
    # 注册调用
    register('ldh', './datas/LFW/images/ldh/img1.jpeg')
    register('ldh', './datas/LFW/images/ldh/img0.jpeg')
    register('zxc', './datas/LFW/images/zxc/img0.jpeg')
    register('zxc', './datas/LFW/images/zxc/img1.jpeg')
    register('zxc', './datas/LFW/images/zxc/img2.jpeg')

    # 人证对比：调用后端接口的时候给定了用户id以及当前图像
    face_compare('ldh', './datas/LFW/images/ldh/img0.jpeg')
    face_compare('zxc', './datas/LFW/images/ldh/img0.jpeg')
    face_compare('zxc', './datas/LFW/images/zxc/img0.jpeg')
    face_compare('zxc', './datas/LFW/images/zxc/img1.jpeg')
    face_compare('zxc', './datas/LFW/images/zxc/img2.jpeg')

    # 人脸检索：调用后端接口的时候，仅给定图像
    face_search('./datas/LFW/images/ldh/img0.jpeg')
    face_search('./datas/LFW/images/ldh/img1.jpeg')
    face_search('./datas/LFW/images/zxc/img0.jpeg')
    face_search('./datas/LFW/images/zxc/img1.jpeg')
    face_search('./datas/LFW/images/zxc/img2.jpeg')
    face_search('./datas/LFW/images/Adrian_Annus/Adrian_Annus_0001.jpg')
