# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/23 20:54
Create User : 19410
Desc : xxx
"""
from typing import Union

import cv2 as cv
import numpy as np
import torch
from PIL import Image

from .data_utils import load_image
from .network import build_network


# noinspection DuplicatedCode
class Predictor(object):
    def __init__(self, model_file='./output/model.pkl'):
        super().__init__()

        # 模型如何持久化的就采用对应方式进行模型恢复
        ckpt = torch.load(model_file, map_location='cpu')
        net_param = ckpt['net_param']  # 模型参数
        class_names = ckpt['class_names']

        net = build_network(
            num_classes=len(class_names),
            in_channels=3
        )

        # 当strict为True的时候，要求给定的参数列表和当前模型的参数列表必须完全匹配
        missing_keys, unexpected_keys = net.load_state_dict(net_param, strict=False)
        print(f"当前模型未进行参数恢复的相关参数名称列表:{missing_keys}")
        print(f"给定参数在当前模型中不存在:{unexpected_keys}")
        if len(missing_keys) > 0:
            raise ValueError(f"模型恢复异常:{missing_keys}")

        self.net = net.eval()
        self.class_names = class_names

    @torch.no_grad()
    def predict(self, img_file: Union[str, Image.Image], new_size=(200, 200)):
        # 2. 和训练采用相同的流程，对待预测的数据进行处理转换
        # 图像加载
        img = load_image(img_file)
        # 图像大小缩放
        img = cv.resize(img, new_size)
        # 转换数据
        # [H,W,C] --> [C,H,W]
        img = np.transpose(img, axes=(2, 0, 1))
        img = img / 255.0  # [0,255] -> [0,1]
        img = img[None]  # [C,H,W] --> [1,C,H,W]
        img = torch.tensor(img, dtype=torch.float32)

        # 3. 调用模型的预测方法(前向过程)获取得到预测结果
        score = self.net(img)
        probs = torch.softmax(score, dim=-1)

        # 4. 后处理转换 --> 在模型预测结果的基础上额外的进行一些数据处理的工作
        pred_idx = torch.argmax(score, dim=-1)[0].item()
        pred_name = self.class_names[pred_idx]
        pred_proba = probs[0][pred_idx].item()

        # PS: 具体返回什么格式的数据，需要和需求方进行确认
        return {
            "prediction_name": pred_name,
            "predict_index": pred_idx,
            "predict_proba": pred_proba
        }

    @torch.no_grad()
    def fetch_features(self, img_file: Union[str, Image.Image], new_size=(200, 200)):
        # 2. 和训练采用相同的流程，对待预测的数据进行处理转换
        # 图像加载
        img = load_image(img_file)
        # 图像大小缩放
        img = cv.resize(img, new_size)
        # 转换数据
        # [H,W,C] --> [C,H,W]
        img = np.transpose(img, axes=(2, 0, 1))
        img = img / 255.0  # [0,255] -> [0,1]
        img = img[None]  # [C,H,W] --> [1,C,H,W]
        img = torch.tensor(img, dtype=torch.float32)

        # 3. 调用模型的预测方法(前向过程)获取得到预测结果
        _, features = self.net(img, return_features=True)

        return features[0]
