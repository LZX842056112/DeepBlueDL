# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/15 17:20
Create User : 19410
Desc : 训练相关的入口代码
"""
import os.path
from typing import List

from ultralytics import YOLO
from ultralytics.engine.results import Results


# noinspection PyTypeChecker
def detect_t0():
    """
    训练方式一：模型迁移训练(除了类别数目改动外，其它任何东西都不改动)
    :return:
    """

    # Load a COCO-pretrained YOLOv8n model
    # 加载一个yolo体系的模型(模型结构+模型参数)
    model = YOLO("configs/yolov8n.pt")

    # Display model information (optional)
    model.info()

    # Train the model on the COCO8 example dataset for 100 epochs
    # https://docs.ultralytics.com/zh/usage/cfg/#train-settings
    data = r"datasets/coco8.yaml"
    data = r"datasets/mnist.yaml"
    results = model.train(
        data=data,
        epochs=10,
        imgsz=640,
        batch=2,
        device="cpu",
        workers=0,
        cos_lr=True,
        mixup=0.3,
        project="./runs/python/detect",
        name=f"{os.path.splitext(os.path.basename(data))[0]}_train"
    )
    print(results)
    print("=" * 100)

    # Run inference with the YOLOv8n model on the 'bus.jpg' image
    model.eval()
    results: List[Results] = model(r"images/bus.jpg")
    print(results)
    results[0].save()


def detect_t1():
    """
    训练方式二：直接从0开始训练(不做任何模型迁移的工作)
    如果结构更改的话：
        情况一：不需要新增模块的情况下，直接修改yaml配置文件即可
        情况二：如果需要新增模块，那么需要更改以下内容：
            a. 在 ultralytics/nn/modules 对应py文件中进行编辑增加class
            b. 在 ultralytics/nn/modules/__init__.py 文件中导入新增的class
            c. 在 ultralytics/nn/tasks.py/parse_model 中增加新模块的解析创建相关代码逻辑
            d. 修改yaml配置文件
    :return:
    """
    # 基于yaml文件解析得到一个模型对象
    ### PS: 给定的yaml文件名称中必须包含尺度字符串: n s l m x；并且使用那个版本，就必须包含对应yolo版本名称字符串，eg: yolov8n
    yaml_path = "./configs/yolov8n_coco8.yaml"  # 实际文件名称是: yolov8_coco8.yaml, 但是给定的文件名必须为: yolov8_coco8.yaml
    model = YOLO(yaml_path)

    # Display model information (optional)
    model.info()

    # https://docs.ultralytics.com/zh/usage/cfg/#train-settings
    data = r"datasets/coco8.yaml"
    results = model.train(
        data=data,
        epochs=3,
        imgsz=640, batch=2, device="cpu",
        workers=0, cos_lr=True, mixup=0.3,
        project="./runs/python/detect",
        name=f"{os.path.splitext(os.path.basename(data))[0]}_train"
    )
    print(results)


def detect_t2():
    """
    训练方式三：结构更改 + 部分参数迁移
    如果结构更改的话：
        情况一：不需要新增模块的情况下，直接修改yaml配置文件即可
        情况二：如果需要新增模块，那么需要更改以下内容：
            a. 在 ultralytics/nn/modules 对应py文件中进行编辑增加class
            b. 在 ultralytics/nn/modules/__init__.py 文件中导入新增的class
            c. 在 ultralytics/nn/tasks.py/parse_model 中增加新模块的解析创建相关代码逻辑
            d. 修改yaml配置文件
    :return:
    """
    # 基于yaml文件解析得到一个模型对象
    ### PS: 给定的yaml文件名称中必须包含尺度字符串: n s l m x；并且使用那个版本，就必须包含对应yolo版本名称字符串，eg: yolov8n
    # yaml_path = "./configs/yolov8n_coco8.yaml"  # 实际文件名称是: yolov8_coco8.yaml, 但是给定的文件名必须为: yolov8_coco8.yaml
    # 实际文件名称是: yolov8_coco8_v1.yaml 利用现有的模块，更加网络结构 --> 参考YOLOv6对Neck部分的特征融合采用BiC结构(类似)
    # yaml_path = "./configs/yolov8n_coco8_v1.yaml"
    # 实际文件名称是: yolov8_coco8_v2.yaml  新增模块 + 利用现有模块，更改网络结构 -> 将下采样更改为MP结构(同时有池化和卷积)
    yaml_path = "./configs/yolov8n_coco8_v2.yaml"
    model = YOLO(yaml_path)
    model.load("./configs/yolov8n.pt")  # 模型参数恢复(迁移的参数恢复)

    # https://docs.ultralytics.com/zh/usage/cfg/#train-settings
    data = r"datasets/coco8.yaml"
    results = model.train(
        data=data,
        epochs=3,
        imgsz=640, batch=2, device="cpu",
        workers=0, cos_lr=True, mixup=0.3,
        project="./runs/python/detect",
        name=f"{os.path.splitext(os.path.basename(data))[0]}_train",
        # pretrained="./configs/yolov8n.pt"
    )
    print(results)


def cls_training():
    """
    分类模型训练
    :return:
    """
    # 构造模型以及模型参数迁移初始化
    yaml_path = "./configs/yolov8n-cls.yaml"
    model = YOLO(yaml_path)
    model.load("./configs/yolov8n-cls.pt")  # 模型参数恢复(迁移的参数恢复)

    # https://docs.ultralytics.com/usage/cfg/#train-settings\
    """
    数据保存结构如下:
    {data}  给定的文件夹
        {data}/train  用于模型训练的数据文件夹
            {data}/train/dog  第一个类别为dog类别
                1.png  第一个类别的第一张图像
                2.png  第一个类别的第二张图像
                ....
            {data}/train/cat 第二个类别为cat类别
                1.png
                2.jpg
                3.png
                .....
            {data}/train/person
                ......
        {data}/val 用于模型评估
            {data}/val/dog
                1.png
                2.png
                ....
            {data}/val/cat
                1.png
                2.jpg
                3.png
                .....
            {data}/val/person
                ......
        {data}/test 可选给定，用于最后的模型效果评估
            {data}/test/dog
                1.png
                2.png
                ....
            {data}/test/cat
                1.png
                2.jpg
                3.png
                .....
            {data}/test/person
                ......
    """
    data = r"D:\datas\17flowers"  # 文件夹下要满足图像分类的格式
    results = model.train(
        data=data,
        epochs=3,
        imgsz=320, batch=2, device="cpu",
        workers=0, cos_lr=True, mixup=0.3,
        project="./runs/python/classify",
        name=f"{os.path.splitext(os.path.basename(data))[0]}_train",
    )
    print(results)


def segment_training():
    model = YOLO(r".\configs\yolov8n-seg.yaml", task="segment")  # 结构做了更改
    model.load(r".\configs\yolov8n-seg.pt")  # 参数恢复/迁移初始化

    # Display model information (optional)
    model.info()

    # Train the model on the COCO8 example dataset for 100 epochs
    # https://docs.ultralytics.com/usage/cfg/#train-settings
    data = r".\datasets\coco8-seg.yaml"  # 80个类别的图像分割
    results = model.train(
        data=data,
        epochs=5,
        imgsz=160, batch=4,
        device="cpu", workers=0,
        cos_lr=True, mixup=0.3,
        project="./runs/python/segment",
        name=os.path.splitext(os.path.basename(data))[0]
    )


def keypoint_training():
    """
    关键点检测在YOLOv8中是先进行目标检测，然后在检测出来的目标中进行关键点坐标位置的检测
    训练数据和目标检查差不多，一行一个检测物体
        cid cx cy w h kp1x kp1y kp1v kp2x kp2y kp2v .....
    :return:
    """
    # 迁移模型
    model = YOLO(model='./configs/yolov8n-pose.yaml')
    model.load('./configs/yolov8n-pose.pt')

    # 训练
    model.train(
        data=r"./datasets/coco8-pose.yaml",
        epochs=20,
        batch=4,  # 4
        imgsz=160,  #
        workers=0,
        device='cpu', degrees=10, mixup=0.3,
    )

def face_keypoint_training():
    # 迁移模型
    model = YOLO(model='./configs/yolov8n-pose.yaml')
    model.load('./configs/yolov8n-pose.pt')

    # 训练
    model.train(
        data=r"./datasets/face.yaml",
        epochs=20,
        batch=4,  # 4
        imgsz=160,  #
        workers=0,
        device='cpu', degrees=10, mixup=0.3,
    )
def obb_training():
    # 迁移模型
    model = YOLO(model='./configs/yolov8n-obb.yaml')
    model.load('./configs/yolov8n-obb.pt')

    # 训练
    model.train(
        data=r"./datasets/dota8.yaml",
        epochs=20,
        batch=4,  # 4
        imgsz=160,  #
        workers=0,
        device='cpu', degrees=10, mixup=0.3,
    )


if __name__ == '__main__':
    # detect_t0()
    # cls_training()
    # segment_training()
    # keypoint_training()
    # obb_training()
    face_keypoint_training()
