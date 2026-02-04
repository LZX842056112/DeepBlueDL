# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/30 14:33
Create User : 19410
Desc : xxx
"""

import os

os.environ['XDG_CACHE_HOME'] = r"D:\huggingface"
os.environ['CACHE_HOME'] = r'D:\huggingface'
os.environ['MODELSCOPE_CACHE'] = r'D:\huggingface\modelscope\hub'


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def interface01():
    # !pip install diffusers
    from modelscope import DDPMPipeline, DDIMPipeline, PNDMPipeline

    model_id = "google/ddpm-cat-256"
    model_id = r"D:\huggingface\modelscope\hub\models\google\ddpm-cat-256"

    # load model and scheduler
    ddpm = DDPMPipeline.from_pretrained(
        model_id
    )  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference

    # run pipeline in inference (sample random noise and denoise)
    image = ddpm().images[0]

    # save image
    image.save("ddpm_generated_image.png")


def interface02():
    # !pip install diffusers
    from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline

    model_id = "google/ddpm-cat-256"
    model_id = r"D:\huggingface\modelscope\hub\models\google\ddpm-cat-256"

    # load model and scheduler
    ddpm: DDPMPipeline = DDPMPipeline.from_pretrained(
        model_id
    )  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
    print(type(ddpm))
    print(ddpm.unet)

    # run pipeline in inference (sample random noise and denoise)
    image = ddpm(num_inference_steps=500).images[0]

    # save image
    image.save("ddpm_generated_image_500.png")


if __name__ == '__main__':
    interface02()
