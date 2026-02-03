# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/9 16:21
Create User : 19410
Desc : 从YOLOv5中迁移过来的部分代码
"""

import cv2
import numpy as np


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), stride=32):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2x1y1wh(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0]  # top left x
    y[:, 1] = x[:, 1]  # top left y
    y[:, 2] = x[:, 2] - x[:, 0]  # w
    y[:, 3] = x[:, 3] - x[:, 1]  # h
    return y


def non_max_suppression(
        prediction,  # numpy的数组对象 [M,4+1+nc]
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=300
):
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    # Apply constraints
    x = prediction[xc]

    if not x.shape[0]:
        return None

    # Compute conf
    x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

    # Box/Mask
    box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)

    i, j = (x[:, 5:] > conf_thres).nonzero()
    x = np.concatenate((box[i], x[i, 5 + j, None], j[:, None]), 1)

    n = x.shape[0]  # number of boxes
    if not n:  # no boxes
        return None
    elif n > max_nms:  # excess boxes
        x = x[x[:, 4].argsort()[::-1][:max_nms]]  # sort by confidence
    else:
        x = x[x[:, 4].argsort()[::-1]]  # sort by confidence

    c = x[:, 5:6] * max_wh  # classes
    boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
    i = cv2.dnn.NMSBoxes(xyxy2x1y1wh(boxes), scores, score_threshold=conf_thres, nms_threshold=iou_thres)
    # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
    if i.shape[0] > max_det:  # limit detections
        i = i[:max_det]
    return x[i]


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(boxes, img0_shape, ratio_pad):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    gain = ratio_pad[0][0]
    pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes
