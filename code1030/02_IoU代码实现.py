# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/30 21:14
Create User : 19410
Desc : xxx
"""
import numpy as np


def iou_v1(box1, box2, xywh=False):
    if xywh:
        # 给定的两个边框是中心点 + 宽度高度
        box1_cx, box1_cy, box1_w, box1_h = box1
        box2_cx, box2_cy, box2_w, box2_h = box2

        box1_x1 = box1_cx - box1_w / 2.0
        box1_x2 = box1_cx + box1_w / 2.0
        box1_y1 = box1_cy - box1_h / 2.0
        box1_y2 = box1_cy + box1_h / 2.0

        box2_x1 = box2_cx - box2_w / 2.0
        box2_x2 = box2_cx + box2_w / 2.0
        box2_y1 = box2_cy - box2_h / 2.0
        box2_y2 = box2_cy + box2_h / 2.0
    else:
        # 给定的两个边框是左上角+右下角坐标
        box1_x1, box1_y1, box1_x2, box1_y2 = box1
        box2_x1, box2_y1, box2_x2, box2_y2 = box2

        box1_w = box1_x2 - box1_x1
        box1_h = box1_y2 - box1_y1
        box2_w = box2_x2 - box2_x1
        box2_h = box2_y2 - box2_y1

    box1_area = box1_w * box1_h
    box2_area = box2_w * box2_h

    # 交叉区域的面积
    inter_w = min(box1_x2, box2_x2) - max(box1_x1, box2_x1)
    inter_w = max(inter_w, 0)
    inter_h = min(box1_y2, box2_y2) - max(box1_y1, box2_y1)
    inter_h = max(inter_h, 0)
    inter_area = inter_w * inter_h

    # iou计算
    iou = inter_area / (box1_area + box2_area - inter_area)

    return iou


def iou_v2(box, xywh=False):
    """
    计算边框之间的IoU
    :param box: [N,4]
    :param xywh: 给定坐标是否是中心点+宽度/高度
    :return:
    """
    if xywh:
        box_cx, box_cy, box_w, box_h = np.split(box, 4, axis=1)
        box_x1 = box_cx - box_w / 2.0
        box_x2 = box_cx + box_w / 2.0
        box_y1 = box_cy - box_h / 2.0
        box_y2 = box_cy + box_h / 2.0
    else:
        # 给定的两个边框是左上角+右下角坐标
        box_x1, box_y1, box_x2, box_y2 = np.split(box, 4, axis=1)
        box_w = box_x2 - box_x1  # [N,1]
        box_h = box_y2 - box_y1  # [N,1]

    box_area = box_w * box_h  # [N,1]

    # 交叉区域的面积
    box1_x1, box2_x1 = np.meshgrid(box_x1, box_x1.T)
    inter_x1 = np.max([box1_x1, box2_x1], axis=0)
    box1_x2, box2_x2 = np.meshgrid(box_x2, box_x2.T)
    inter_x2 = np.min([box1_x2, box2_x2], axis=0)
    inter_w = inter_x2 - inter_x1
    inter_w = np.where(inter_w <= 0, 0, inter_w)

    box1_y1, box2_y1 = np.meshgrid(box_y1, box_y1.T)
    inter_y1 = np.max([box1_y1, box2_y1], axis=0)
    box1_y2, box2_y2 = np.meshgrid(box_y2, box_y2.T)
    inter_y2 = np.min([box1_y2, box2_y2], axis=0)
    inter_h = inter_y2 - inter_y1
    inter_h = np.where(inter_h <= 0, 0, inter_h)

    inter_area = inter_w * inter_h # [N,N]

    # iou计算
    iou = inter_area / (box_area + box_area.T - inter_area)

    return iou


def t0():
    box1 = [150, 200, 250, 300]
    box2 = [180, 230, 250, 400]
    print(iou_v1(box1, box2))
    print(iou_v1(box1, box2, xywh=True))


def t1():
    box = np.asarray([
        [150, 200, 250, 300],
        [180, 230, 250, 400],
        [175, 258, 200, 300]
    ])
    iou = iou_v2(box, xywh=True)
    print(iou)


if __name__ == '__main__':
    t1()
    # 0.28823529411764703
    # 0.6055045871559633
