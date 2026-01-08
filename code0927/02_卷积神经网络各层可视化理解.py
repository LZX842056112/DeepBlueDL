# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/9/27 15:02
Create User : 19410
Desc : xxx
"""

import os

import torch
from PIL import Image  # pip install pillow==11.1.0
from torchvision import models, transforms, utils


def t0():
    os.makedirs("./output", exist_ok=True)
    # 当给定weights=models.VGG16_Weights.DEFAULT或者pretrained=True的时候，就会从网络上下载训练好的模型参数，并完成参数恢复操作
    # 下载好的模型默认会放到当用户根目录下: ~\.cache\torch\hub\checkpoints；如果想更改默认的保存路径，可以考虑配置一个环境变量: XDG_CACHE_HOME
    # 下载好的模型(vgg16)默认为基于ImageNet数据集训练的一个通用开源数据集
    # ImageNet总共有1000个类别: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a#file-imagenet1000_clsid_to_human-txt
    # net = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    net = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
    # net = models.vgg16()
    net = net.eval()
    print(net)

    # 加载图像
    # path = "../datas/小猫.jpg"
    # path = "../datas/小猫2.jpg"
    # path = "../datas/ssd/images/小狗2.png"
    path = "../code1026/datas/ssd/images/飞机2.jpg"
    # img: Image.Image = Image.open("../datas/小狗.png")
    img: Image.Image = Image.open(path)
    print(img.size)
    # img = img.resize((300, 200))  # w=300, h=200
    img = img.resize((800, 500))
    img = img.convert("RGB")
    print(img.size)

    # 图像转换为tensor对象
    compose = transforms.Compose(
        transforms=[
            transforms.ToTensor()
        ]
    )
    # [C,H,W] C一般表示通道，也就是有多少类特征/多少个角度来描述图像特征；H表示图像的高度，有多少个像素点；W表示图像的宽度，有多少个像素点
    img_tensor = compose(img)
    print(img_tensor.shape)

    # 模型输入一般要求是批次的概念，也就是一次输入多个图像 ≥1
    in_feature = img_tensor[None]  # [C,H,W] -> [1,C,H,W]
    out_score = net(in_feature)  # [1,1000] 1是和输入中的批次大小一致，1000是模型的类别数目
    print(out_score.shape)

    # 提取预测概率最大的前5个类别的id以及概率
    out_prob = torch.softmax(out_score, dim=-1)
    topk_prob_values, topk_class_indices = torch.topk(out_prob, k=5, dim=-1)
    print(f"预测类别概率:{topk_prob_values}")
    print(f"预测类别id:{topk_class_indices}")

    # =========================================================

    fp_output_dir = os.path.join("./output", os.path.splitext(os.path.basename(path))[0])
    os.makedirs(fp_output_dir, exist_ok=True)
    print(f"输出文件夹:{fp_output_dir}")

    # 输出各个通道的特征图像
    layer_out_feature = net.features[0](in_feature)  # 得到当前层的输出特征矩阵 [N,C,H,W]
    print(layer_out_feature.shape)  # [1, 64, 500, 800]  1个图像由64个类别的feature map
    utils.save_image(
        tensor=torch.transpose(layer_out_feature, 0, 1),  # [1, 64, 500, 800] -> [64, 1, 500, 800] # 64个图像，每个图像一个通道
        fp=f"{fp_output_dir}/layer_1.png"
    )
    weight = net.features[0].weight
    bias = net.features[0].bias
    print(weight.shape)
    print(bias.shape)
    print(weight[0])
    print(bias[0])

    in_feature = layer_out_feature  # 将上一层的输出作为当前层的输入
    layer_out_feature = net.features[1](in_feature)  # 得到当前层的输出特征矩阵 [N,C,H,W]
    print(layer_out_feature.shape)  # [1, 64, 500, 800]
    utils.save_image(
        tensor=torch.transpose(layer_out_feature, 0, 1),  # [1, 64, 500, 800] -> [64, 1, 500, 800]
        fp=f"{fp_output_dir}/layer_2.png"
    )

    in_feature = layer_out_feature  # 将上一层的输出作为当前层的输入
    layer_out_feature = net.features[2](in_feature)  # 得到当前层的输出特征矩阵 [N,C,H,W]
    print(layer_out_feature.shape)
    utils.save_image(
        tensor=torch.transpose(layer_out_feature, 0, 1),
        fp=f"{fp_output_dir}/layer_3.png"
    )

    in_feature = layer_out_feature  # 将上一层的输出作为当前层的输入
    layer_out_feature = net.features[3](in_feature)  # 得到当前层的输出特征矩阵 [N,C,H,W]
    print(layer_out_feature.shape)
    utils.save_image(
        tensor=torch.transpose(layer_out_feature, 0, 1),
        fp=f"{fp_output_dir}/layer_4.png"
    )

    in_feature = layer_out_feature  # 将上一层的输出作为当前层的输入
    layer_out_feature = net.features[4](in_feature)  # 得到当前层的输出特征矩阵 [N,C,H,W]
    print(layer_out_feature.shape)
    utils.save_image(
        tensor=torch.transpose(layer_out_feature, 0, 1),
        fp=f"{fp_output_dir}/layer_5.png"
    )

    # 前面整个features层的特征提取图
    in_feature = img_tensor[None]
    layer_out_feature = net.features(in_feature)  # 得到当前层的输出特征矩阵 [N,C,H,W]
    print(layer_out_feature.shape)
    utils.save_image(
        tensor=torch.transpose(layer_out_feature, 0, 1),
        fp=f"{fp_output_dir}/cat_layer_all.png",
        pad_value=0.5
    )

    # TODO: 将VGG的所有层的feature map形成图像均输出，输出到独立的文件夹中：一层一张图片、支持更改输入图像、输出文件夹、输出文件名称格式；如果模型结构发生变化，支持快速适配
    print("成功!!")


def t1():
    net = models.mobilenet_v2()
    print(net)

if __name__ == '__main__':
    t1()

