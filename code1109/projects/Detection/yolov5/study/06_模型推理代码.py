# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/9 14:13
Create User : 19410
Desc : xxx
"""
from pathlib import Path

import cv2
import torch

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes


def t0():
    # weights = r'../runs/train/coco128/weights/best.pt'
    weights = r'../runs/train/coco128/weights/best.torchscript'
    weights = r'../runs/train/coco128/weights/best.onnx'
    device = torch.device("cpu")
    dnn = False
    data = "../data/coco128_copy.yaml"
    half = False
    new_shape = 640

    # 模型恢复
    # dnn=True or False: True表示使用OpenCV作为底层的推理框架 --> 仅支持ONNX格式并且固定shape尺度的输入(不支持动态的shape输入)
    # fp16=True or False: 是否使用半精度进行推理
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    model.eval()

    img_file = "../data/images/dog.png"
    img = cv2.imread(img_file)  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ori_shape = img.shape[:2]
    # 自适应的最小图像缩放
    # im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32
    img, ratio, pad = letterbox(img, new_shape, auto=True, scaleup=False)
    print("图像大小:", img.shape)
    print(ratio)
    print(pad)

    # 前向推理
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = img[None]  # CHW to NCHW
    img = torch.tensor(img, dtype=torch.float32)
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    augment = False  # 推理使用采用数据增强：推理的时候，对图像进行resize缩放操作
    # visualize = Path("./output/v1")  # 可视化每一层的feature map
    # visualize.mkdir(parents=True, exist_ok=True)
    visualize = None
    preds = model(img, augment=augment, visualize=visualize)
    print(preds[0].shape)

    preds = non_max_suppression(
        preds,  # 模型预测信息 [N, M, 4 + 1 + nc] N个图像，每个图像M个预测边框，每个边框4+1+nc个预测值
        conf_thres=0.3,  # 分类概率阈值
        iou_thres=0.6,  # NMS IoU阈值
        labels=[],
        multi_label=True,  # 是否允许一个预测边框属于多个类别
        agnostic=False,
        max_det=300  # 一个图像最多允许返回多少个预测边框
    )
    print(preds)

    # 图像尺度还原
    for si, pred in enumerate(preds):
        scale_boxes(img[si].shape[1:], pred[:, :4], ori_shape, (ratio, pad))

    print(preds)


def t1():
    expect_img_size = 320

    img1 = cv2.imread(r"../data/images/dog.png")
    h1, w1 = img1.shape[:2]
    img2, ratio, pad = letterbox(
        im=img1.copy(),  # 操作的图像
        new_shape=expect_img_size,  # 希望输出的图像尺度大小
        color=(114, 114, 114),  # 填充像素值
        auto=True,  # 是否做自适应的图像缩放 --> 自适应最小填充
        scaleFill=False,  # auto=False，scaleFill=True的时候表示进行不等比例缩放
        scaleup=False,  # False表示仅支持缩小，不支持放大；True表示支持图像放大
        stride=32  # 模型的缩放比例，会让letterbox输出的图像大小能够整除stride
    )
    h2, w2 = img2.shape[:2]

    print(h1, w1, h2, w2, ratio, pad)
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()


def t2():
    # python export.py --weights ./runs/train/coco128/weights/best.pt --include onnx torchscript
    # PS: 使用OpenCV加载ONNX的时候，要求给定的模型文件输入、输出的shape必须是固定的
    weights = r'../runs/train/coco128/weights/best.onnx'
    new_shape = 640

    # 模型恢复
    net = cv2.dnn.readNetFromONNX(weights)

    img_file = "../data/images/dog.png"
    img = cv2.imread(img_file)  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ori_shape = img.shape[:2]
    # 自适应的最小图像缩放
    # im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32
    img, ratio, pad = letterbox(img, new_shape, auto=False, scaleup=True)
    print("图像大小:", img.shape)
    print(ratio)
    print(pad)

    # 前向推理
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = img[None]  # CHW to NCHW
    img = torch.tensor(img, dtype=torch.float32)
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    im = img.cpu().numpy()  # torch to numpy
    net.setInput(im)
    y = net.forward()
    preds = torch.tensor(y).to(dtype=torch.float32)

    # NMS处理：类别过滤 + top-n + NMS + top-m
    preds = non_max_suppression(
        preds,  # 模型预测信息 [N, M, 4 + 1 + nc] N个图像，每个图像M个预测边框，每个边框4+1+nc个预测值
        conf_thres=0.3,  # 分类概率阈值
        iou_thres=0.6,  # NMS IoU阈值
        labels=[],
        multi_label=True,  # 是否允许一个预测边框属于多个类别
        agnostic=False,
        max_det=300  # 一个图像最多允许返回多少个预测边框
    )
    print(preds)

    # 图像尺度还原
    for si, pred in enumerate(preds):
        scale_boxes(img[si].shape[1:], pred[:, :4], ori_shape, (ratio, pad))

    print(preds)


if __name__ == '__main__':
    t2()
