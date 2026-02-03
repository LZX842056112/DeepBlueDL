# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/30 14:33
Create User : 19410
Desc : 示例代码，演示如何使用 ModelScope 和 Diffusers 库加载 DDPM 模型并生成图像。
"""

import os

# 设置缓存目录，用于存储 Hugging Face 和 ModelScope 的模型文件
os.environ['XDG_CACHE_HOME'] = r"D:\Projects\DeepBule\d2l-zh\code1130\huggingface"
os.environ['CACHE_HOME'] = r'D:\Projects\DeepBule\d2l-zh\code1130\huggingface'
os.environ['MODELSCOPE_CACHE'] = r'D:\Projects\DeepBule\d2l-zh\code1130\modelscope\hub'

# 设置 Hugging Face 镜像地址，加速模型下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def interface01():
    """
    使用 ModelScope 库加载 DDPM 模型并生成图像。
    """
    # 从 ModelScope 导入 DDPM 相关的 Pipeline 类
    from modelscope import DDPMPipeline, DDIMPipeline, PNDMPipeline

    # 指定模型 ID（Hugging Face 上的模型标识符）
    model_id = "google/ddpm-cat-256"
    # 或者指定本地模型路径
    model_id = r"D:\Projects\DeepBule\d2l-zh\code1130\modelscope\hub\models\google\ddpm-cat-256"

    # 加载预训练模型和调度器
    ddpm = DDPMPipeline.from_pretrained(
        model_id
    )  # 可替换为 DDIMPipeline 或 PNDMPipeline 以实现更快推理

    # 运行推理过程：采样随机噪声并逐步去噪生成图像
    image = ddpm().images[0]

    # 保存生成的图像
    image.save("ddpm_generated_image.png")


def interface02():
    """
    使用 Diffusers 库加载 DDPM 模型并生成图像。
    """
    # 从 Diffusers 导入 DDPM 相关的 Pipeline 类
    from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline

    # 指定模型 ID（Hugging Face 上的模型标识符）
    model_id = "google/ddpm-cat-256"
    # 或者指定本地模型路径
    model_id = r"D:\Projects\DeepBule\d2l-zh\code1130\modelscope\hub\models\google\ddpm-cat-256"

    # 加载预训练模型和调度器
    ddpm: DDPMPipeline = DDPMPipeline.from_pretrained(
        model_id
    )  # 可替换为 DDIMPipeline 或 PNDMPipeline 以实现更快推理

    # 打印模型类型和 UNet 结构信息
    print(type(ddpm))
    print(ddpm.unet)

    # 运行推理过程：采样随机噪声并逐步去噪生成图像
    image = ddpm(num_inference_steps=20).images[0]

    # 保存生成的图像
    image.save("ddpm_generated_image_20.png")


if __name__ == '__main__':
    # interface01()
    interface02()