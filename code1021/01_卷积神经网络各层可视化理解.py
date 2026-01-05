# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/9/27 15:02
Create User : 19410
Desc : xxx
"""

import os

import torch
import torch.nn as nn

from PIL import Image  # pip install pillow==11.1.0
from torchvision import models, transforms, utils


def build_hook_func(module_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    def hook_func(m, m_args, m_outputs):
        """
        :param m: 当前模块对象
        :param m_args: 当前模块的当前forward的入参，tuple类型
        :param m_outputs: 当前模块的当前forward方法的返回结果，也就是forward的返回值
        :return:
        """
        show_imgs = m_outputs[0:1]  # 仅对当前批次的第一张图像进行可视化 [1,C,H,W]
        show_imgs = torch.transpose(show_imgs, 0, 1)  # [C,1,H,W] 将每个通道看成一个图像
        utils.save_image(
            tensor=show_imgs,
            fp=f"{output_dir}/{module_name}.png",
            pad_value=0.5
        )
        return None

    return hook_func


def t0():
    os.makedirs("./output", exist_ok=True)
    # 当给定weights=models.VGG16_Weights.DEFAULT或者pretrained=True的时候，就会从网络上下载训练好的模型参数，并完成参数恢复操作
    # 下载好的模型默认会放到当用户根目录下: ~\.cache\torch\hub\checkpoints；如果想更改默认的保存路径，可以考虑配置一个环境变量: XDG_CACHE_HOME
    # 下载好的模型(vgg16)默认为基于ImageNet数据集训练的一个通用开源数据集
    # ImageNet总共有1000个类别: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a#file-imagenet1000_clsid_to_human-txt
    # net = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    # net = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
    # net = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    net = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    # net = models.vgg16()
    net = net.eval()
    print(net)

    # 加载图像
    path = "dog.jpeg"
    # path = "../datas/小猫2.jpg"
    # path = "../datas/ssd/images/小狗2.png"
    # path = "../datas/ssd/images/飞机2.jpg"
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

    # 在卷积、BN、池化层上增加hook function
    hook_handles = []
    show_feature_dir = os.path.join("./output", net.__class__.__name__, os.path.splitext(os.path.basename(path))[0])
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d):
            _hook_fn = build_hook_func(
                module_name=f"{name}_conv",
                output_dir=show_feature_dir
            )
            handle = module.register_forward_hook(hook=_hook_fn)
            hook_handles.append(handle)
        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d)):
            _hook_fn = build_hook_func(
                module_name=f"{name}_pool",
                output_dir=show_feature_dir
            )
            handle = module.register_forward_hook(hook=_hook_fn)
            hook_handles.append(handle)
        elif isinstance(module, nn.BatchNorm2d):
            _hook_fn = build_hook_func(
                module_name=f"{name}_bn",
                output_dir=show_feature_dir
            )
            handle = module.register_forward_hook(hook=_hook_fn)
            hook_handles.append(handle)

    # 模型输入一般要求是批次的概念，也就是一次输入多个图像 ≥1
    in_feature = img_tensor[None]  # [C,H,W] -> [1,C,H,W]
    out_score = net(in_feature)  # [1,1000] 1是和输入中的批次大小一致，1000是模型的类别数目
    print(out_score.shape)

    # PS: 当不需要执行hook相关代码的时候，一定要进行删除的操作
    for handle in hook_handles:
        handle.remove()


if __name__ == '__main__':
    t0()
