# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/9 16:16
Create User : 19410
Desc : YOLOv5的模型预测器方法
"""
import cv2 as cv
import numpy as np

from yolov5_deploy_utils import letterbox, non_max_suppression, scale_boxes


class YOLOv5Predictor(object):
    def __init__(self, model_file, img_size=640, conf_thres=0.3, iou_thres=0.6, max_det=300):
        super().__init__()
        self.net = cv.dnn.readNetFromONNX(model_file)
        self.img_size = img_size

        self.conf_thres = max(min(conf_thres, 0.9), 0.01)
        self.iou_thres = max(min(iou_thres, 0.9), 0.01)
        self.max_det = max(max_det, 1)

        # Checks
        assert 0 <= self.conf_thres <= 1, f'Invalid Confidence threshold {self.conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= self.iou_thres <= 1, f'Invalid IoU {self.iou_thres}, valid values are between 0.0 and 1.0'
        print(f"模型恢复完成: {model_file}")

    def predict(self, img):
        """
        基于传递过来的img图像进行推理预测（目标检测）
        :param img: 图像对象，要求格式必须为NumPy 或者 OpenCV的格式(RGB格式)
        :return: list[dict]
        """
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
            y[0],  # 模型预测信息 [M, 4 + 1 + nc] 一个图像M个预测边框，每个边框4+1+nc个预测值
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
