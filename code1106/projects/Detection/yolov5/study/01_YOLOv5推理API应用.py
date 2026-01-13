# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/1 14:23
Create User : 19410
Desc : xxx
"""
import os

import torch


def t0():
    # torch.hub ---> 主要作用是支持直接从github上下载代码以及模型文件，加载恢复进行预测 --> 只要求你加载的模型文件有hubconf.py文件
    # Load a YOLOv5 model (options: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)
    model = torch.hub.load(
        repo_or_dir="ultralytics/yolov5:v7.0",  # 给定github上的项目名称或者本地文件夹路径
        model="yolov5s",  # 给定模型文件，其实就是hubconf.py中的方法名
        source='github'  # 加载的代码/模型来源；可选: github、local
    )
    # Define the input image source (URL, local file, PIL image, OpenCV frame, numpy array, or list)
    img = "https://ultralytics.com/images/zidane.jpg"  # Example image
    img = r"../data/images/bus.jpg"

    # Perform inference (handles batching, resizing, normalization automatically) 模型的推理预测
    results = model(img)

    # Process the results (options: .print(), .show(), .save(), .crop(), .pandas())
    results.print()  # Print results to console
    results.show()  # Display results in a window
    results.save()  # Save results to runs/detect/exp


def t1():
    # torch.hub ---> 主要作用是支持直接从github上下载代码以及模型文件，加载恢复进行预测 --> 只要求你加载的模型文件有hubconf.py文件
    # Load a YOLOv5 model (options: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)
    print(f"绝对路径位置为:{os.path.abspath('..')}")
    model = torch.hub.load(
        repo_or_dir="..",  # 给定github上的项目名称或者本地文件夹路径
        model="yolov5s",  # 给定模型文件，其实就是hubconf.py中的方法名
        source='local'  # 加载的代码/模型来源；可选: github、local
    )
    # Define the input image source (URL, local file, PIL image, OpenCV frame, numpy array, or list)
    img = "https://ultralytics.com/images/zidane.jpg"  # Example image
    img = r"../data/images/bus.jpg"

    # Perform inference (handles batching, resizing, normalization automatically) 模型的推理预测
    results = model(img)

    # Process the results (options: .print(), .show(), .save(), .crop(), .pandas())
    results.print()  # Print results to console
    results.show()  # Display results in a window
    results.save()  # Save results to runs/detect/exp


def t2():
    import sys
    import os


    sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

    print(sys.path)

    # 下列import导包成功的前提条件是：hubconf和models所在的文件夹必须在sys.path环境变量中
    from hubconf import custom
    from models.common import Detections

    # 模型恢复
    model = custom(
        # path='./yolov5s.pt'
        path=r"..\runs\train\coco128\weights\best.pt"
    )

    # 模型推理应用
    img = r"../data/images/bus.jpg"
    results: Detections = model(img)

    results.print()  # Print results to console
    results.show()  # Display results in a window
    print(results.xywhn)


if __name__ == '__main__':
    # t0()
    # t1()
    t2()
