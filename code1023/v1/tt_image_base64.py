# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/23 21:25
Create User : 19410
Desc : xxx
"""

import base64
from io import BytesIO

from PIL import Image


def image_base64(image_path, print_data=False):
    with open(image_path, "rb") as reader:
        img_content = reader.read()  # 加载图像的所有二进制数据
        img_base64_content = base64.b64encode(img_content)  # 编码
        if print_data:
            print(img_base64_content)
        return img_base64_content


def base64_image_str_2_image(img_str):
    img_content = base64.b64decode(img_str)  # 解码还原
    image = Image.open(BytesIO(img_content))
    image = image.convert("RGB")
    return image


if __name__ == '__main__':
    b64str = image_base64(
        image_path=r"../../code1026/datas/c1_image_0016.jpg",
        print_data=True
    )

    img = base64_image_str_2_image(b64str)
    img.show()

    import numpy as np
    np_img = np.asarray(img, dtype=np.uint8)
    print(np_img.shape)

    import cv2 as cv

    cv_img = cv.cvtColor(np_img, cv.COLOR_RGB2BGR)
    cv.imshow('img', cv_img)
    cv.waitKey()
    cv.destroyAllWindows()
