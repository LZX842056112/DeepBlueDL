# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/30 16:59
Create User : 19410
Desc : xxx
"""
import os

os.environ['XDG_CACHE_HOME'] = r"D:\Projects\DeepBule\d2l-zh\code1130\huggingface"
os.environ['CACHE_HOME'] = r'D:\Projects\DeepBule\d2l-zh\code1130\huggingface'
os.environ['MODELSCOPE_CACHE'] = r'D:\Projects\DeepBule\d2l-zh\code1130\modelscope\hub'

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def interface01():
    import torch
    from diffusers import StableDiffusionPipeline

    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]

    image.save("astronaut_rides_horse.png")


def interface02():
    import torch
    from modelscope import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "AI-ModelScope/stable-diffusion-3.5-medium",
        # torch_dtype=torch.bfloat16
    )
    pipe = pipe.to("cpu")

    image = pipe(
        "A capybara holding a sign that reads Hello World",
        num_inference_steps=40,
        guidance_scale=4.5,
    ).images[0]
    image.save("capybara.png")


if __name__ == '__main__':
    # interface01()
    interface02()
