# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/16 17:01
Create User : 19410
Desc : 预测器
"""
from typing import List

import cv2
import numpy as np



class BaseYOLOV8Predictor(object):
    def predict(self, img):
        raise NotImplementedError("子类实现!")


class YOLOV8Predictor(BaseYOLOV8Predictor):
    def __init__(self, model_file: str, img_size=320, conf_thres=0.3, iou_thres=0.4, max_det=300, dnn=True):
        super().__init__()
        from ultralytics import YOLO

        self.model = YOLO(model=model_file, task="detect", verbose=True)
        if model_file.endswith(".pt"):
            self.model.fuse()

        self.dnn = dnn and model_file.endswith(".onnx")
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = img_size
        self.conf_thres = max(min(conf_thres, 0.9), 0.01)
        self.iou_thres = max(min(iou_thres, 0.9), 0.01)
        self.max_det = max(max_det, 1)

        # Checks
        assert 0 < self.conf_thres < 1, f'Invalid Confidence threshold {self.conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 < self.iou_thres < 1, f'Invalid IoU {self.iou_thres}, valid values are between 0.0 and 1.0'

    def predict(self, img):
        """
        预测过程
        :param img: np.ndarray 或者 PIL.Image对象， 图像格式均为RGB格式
        :return:
        """
        from ultralytics.engine.results import Results
        import torch

        if isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        with torch.no_grad():
            results: List[Results] = self.model(
                img,
                conf=self.conf_thres,  # 概率阈值
                iou=self.iou_thres,  # NMS处理过程中的IoU阈值
                imgsz=self.img_size,  # 图像大小
                max_det=self.max_det,  # 一张图像最多允许的输出边框数目
                dnn=self.dnn
            )
            pred = results[0]

        # 将预测结果转换为我们期望的输出结果
        result = []
        for xyxy, conf, cls in zip(
                pred.boxes.xyxy.numpy(),
                pred.boxes.conf.numpy(),
                pred.boxes.cls.numpy()
        ):
            result.append({
                'x1': int(xyxy[0].round(0)),
                'y1': int(xyxy[1].round(0)),
                'x2': int(xyxy[2].round(0)),
                'y2': int(xyxy[3].round(0)),
                'conf': float(conf.round(3)),
                'id': int(cls),
                # 'clsss_name': self.id2name[int(pred[5])]
            })
        return result


class NewYOLOV8Predictor(BaseYOLOV8Predictor):
    def __init__(self, model_file: str, img_size=320, conf_thres=0.3, iou_thres=0.4, max_det=300, dnn=True):
        super().__init__()
        self.net = cv2.dnn.readNetFromONNX(model_file)

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = img_size
        self.conf_thres = max(min(conf_thres, 0.9), 0.01)
        self.iou_thres = max(min(iou_thres, 0.9), 0.01)
        self.max_det = max(max_det, 1)

        # Checks
        assert 0 < self.conf_thres < 1, f'Invalid Confidence threshold {self.conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 < self.iou_thres < 1, f'Invalid IoU {self.iou_thres}, valid values are between 0.0 and 1.0'

    def predict(self, img):
        """
        预测过程
        :param img: np.ndarray 或者 PIL.Image对象， 图像格式均为RGB格式
        :return:
        """
        from yolov8_deploy_utils import letterbox, non_max_suppression, scale_boxes

        ori_shape = img.shape[:2]

        # 图像缩放
        img, ratio, pad = letterbox(img, self.img_size)

        # 前向推理
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = img[None]  # CHW to NCHW
        img = np.asarray(img, dtype=np.float32) / 255.0  # 0 - 255 to 0.0 - 1.0
        self.net.setInput(img)
        y = self.net.forward()

        # NMS处理：类别过滤 + top-n + NMS + top-m
        pred = non_max_suppression(
            y[0].T,  # 模型预测信息 [M, 4 + nc] 一个图像M个预测边框，每个边框4+1+nc个预测值
            conf_thres=self.conf_thres,  # 分类概率阈值
            iou_thres=self.iou_thres,  # NMS IoU阈值
            max_det=self.max_det  # 一个图像最多允许返回多少个预测边框
        )
        if pred is None:
            return []

        # 图像尺度还原
        scale_boxes(pred[:, :4], ori_shape, (ratio, pad))

        # 将preds转换为预测结果list[dict]的形式
        result = []
        for pred in pred:
            result.append({
                'x1': int(pred[0].round(0)),
                'y1': int(pred[1].round(0)),
                'x2': int(pred[2].round(0)),
                'y2': int(pred[3].round(0)),
                'conf': float(pred[4].round(3)),
                'id': int(pred[5]),
                # 'clsss_name': self.id2name[int(pred[5])]
            })
        return result
