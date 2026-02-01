# -*- coding: utf-8 -*-
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys

print(f'将下列路径添加到环境变量中 {os.path.abspath(os.path.join(__file__, "..", ".."))}')
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

if __name__ == '__main__':
    # Flask框架搭建的后端API
    # 基于faiss框架提供的向量检索服务：pip install faiss-cpu==1.8.0
    from face_web_api.app import app

    app.run(host="0.0.0.0", port=9998)
