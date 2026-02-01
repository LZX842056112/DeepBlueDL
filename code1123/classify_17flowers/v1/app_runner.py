# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/23 21:34
Create User : 19410
Desc : xxx

Linux安装OpenCV异常：
    ImportError: libGL.so.1: cannot open shared object file: No such file or directory
    解决办法：
        使用opencv-python-headless替代OpenCV
        pip install opencv-python-headless==4.5.5.64

"""


if __name__ == '__main__':
    import os

    if 'MODEL_FILE' not in os.environ:
        os.environ['MODEL_FILE'] = os.path.abspath('./deploy/best.pkl')

    from img_classify.flask_app import app

    app.run(
        host="0.0.0.0",
        port=9002
    )
