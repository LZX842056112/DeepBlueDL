# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/4 21:41
Create User : 19410
Desc : xxx
"""

import torch
from PIL import Image
from torchvision import models
from torchvision.transforms import transforms


def t0():
    net = models.detection.ssd300_vgg16(
        weights=models.detection.SSD300_VGG16_Weights.DEFAULT,  # SSD部分的参数
        weights_backbone=models.VGG16_Weights.DEFAULT  # 基础网络部分的参数
    )
    print(net)


@torch.no_grad()
def interface():
    ts = transforms.Compose([
        transforms.ToTensor()
    ])
    net = models.detection.ssd300_vgg16(
        weights=models.detection.SSD300_VGG16_Weights.DEFAULT,  # SSD部分的参数
        weights_backbone=models.VGG16_Weights.DEFAULT,  # 基础网络部分的参数
        score_thresh=0.3
    )
    net.eval()

    images = [
        torch.rand(3, 300, 400),
        ts(Image.open("dog.jpeg").convert("RGB")),
        ts(Image.open("cat.jpg").convert("RGB"))
    ]
    # images 是一个list，内部是[C,H,W]结构的tensor对象，每个tensor的H、W可以是不一样的，list中的tensor数量就是图像image_number
    # predictions 是一个list，里面包含的每个样本的预测结果，结果类型为dict
    # ### boxes: [M,4]的结构，表示预测包含M个边框的x1y1x2y2的坐标信息；
    # ### labels: [M]的结构，表示预测M个边框的对应预测类别id，这个类别id就是COCO类别id
    # ### scores: [M]的结构，表示预测M个边框属于对应类别的概率值
    predictions = net(images)
    print(predictions)

    # 静态转换，方便通过网页进行结构的查看
    # PS: 要求安装onnx  pip install onnx==1.17.0
    # https://netron.app/
    torch.onnx.export(net, (images,), "ssd.onnx", opset_version=12)


def training():
    ts = transforms.Compose([
        transforms.ToTensor()
    ])
    net = models.detection.ssd300_vgg16(
        weights=models.detection.SSD300_VGG16_Weights.DEFAULT,  # SSD部分的参数
        weights_backbone=models.VGG16_Weights.DEFAULT  # 基础网络部分的参数
    )

    images = [
        torch.rand(3, 300, 400),
        ts(Image.open("dog.jpeg").convert("RGB")),
        ts(Image.open("cat.jpg").convert("RGB"))
    ]
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
    # images 是一个list，内部是[C,H,W]结构的tensor对象，每个tensor的H、W可以是不一样的，list中的tensor数量就是图像image_number
    # targets 是一个list，内部就是每个图像对应的目标属性信息dict对象，也就是len(images) == len(targets);
    # ### targets中每个dict中包含两个值，boxes和labels，M表示当前图像中真实边框的数量
    # ### boxes 给定当前图像的真实边框坐标tensor, shape形状为:[M,4], 坐标顺序[x1,y1,x2,y2]
    # ### labels 给定图像的真实边框类别id tensor，shape形状为: [M]
    # output包含SSD的分类和回归损失, 是一个dict[str, tensor]的结构
    # ## bbox_regression: 回归损失
    # ## classification: 分类损失
    output = net(images, targets)
    print(output)


if __name__ == '__main__':
    # interface()
    training()
