# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/4 20:13
Create User : 19410
Desc : xxx
"""
import torch
from PIL import Image
from torchvision import transforms
from torchvision import models


def t0():
    net = models.detection.fasterrcnn_resnet50_fpn()
    print(net)


# 禁用梯度计算
@torch.no_grad()
def interface():
    ts = transforms.Compose([
        transforms.ToTensor()
    ])
    net = models.detection.fasterrcnn_resnet50_fpn(
        # 加载预训练权重 ， COCO 预训练权重
        weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        # 检测的阈值，低于这个阈值的边框会被忽略
        box_score_thresh=0.3
    )
    net.eval()

    images = [
        # 每个图像张量形状为 [C, H, W] (通道数, 高度, 宽度)
        torch.rand(3, 300, 400),
        ts(Image.open("cat.jpg").convert("RGB")),
        ts(Image.open("dog.jpeg").convert("RGB"))
    ]
    # images 是一个list，内部是[C,H,W]结构的tensor对象，每个tensor的H、W可以是不一样的，list中的tensor数量就是图像image_number
    # predictions 是一个list，里面包含的每个样本的预测结果，结果类型为dict
    # ### boxes: [M,4]的结构，表示预测包含M个边框的x1y1x2y2的坐标信息；
    # ### labels: [M]的结构，表示预测M个边框的对应预测类别id，这个类别id就是COCO类别id
    # ### scores: [M]的结构，表示预测M个边框属于对应类别的概率值
    predictions = net(images)
    print(predictions)


def training():
    ts = transforms.Compose([
        transforms.ToTensor()
    ])
    net = models.detection.fasterrcnn_resnet50_fpn(
        weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        box_score_thresh=0.3
    )

    # images 是一个list，内部是[C,H,W]结构的tensor对象，每个tensor的H、W可以是不一样的，list中的tensor数量就是图像image_number
    images = [
        torch.rand(3, 300, 400),
        ts(Image.open("cat.jpg").convert("RGB")),
        ts(Image.open("dog.jpeg").convert("RGB"))
    ]
    # targets 是一个list，内部就是每个图像对应的目标属性信息dict对象，也就是len(images) == len(targets);
    # 包含真实标注的列表
    targets = [
        {
            'boxes': torch.tensor([]).reshape((0, 4)),
            'labels': torch.tensor([], dtype=torch.int64)
        },
        {
            'boxes': torch.tensor([[183, 43, 654, 452]]),
            'labels': torch.tensor([18]),
        },
        {
            'boxes': torch.tensor([[222, 33, 776, 294],
                                   [43, 523, 1061, 720],
                                   [172, 209, 770, 714]]),
            'labels': torch.tensor([17, 67, 62])
        }
    ]
    # ### targets中每个dict中包含两个值，boxes和labels，M表示当前图像中真实边框的数量
    # ### boxes 给定当前图像的真实边框坐标tensor, shape形状为:[M,4], 坐标顺序[x1,y1,x2,y2]
    # ### labels 给定图像的真实边框类别id tensor，shape形状为: [M]
    # output包含RPN和Fast R-CNN的分类和回归损失, 是一个dict[str, tensor]的结构
    # ## loss_classifier: Fast R-CNN的分类损失
    # ## loss_box_reg: Fast R-CNN的回归损失
    # ## loss_objectness: RPN的分类损失
    # ## loss_rpn_box_reg: RPN的回归损失
    output = net(images, targets)
    print(output)


if __name__ == '__main__':
    # interface()
    training()
