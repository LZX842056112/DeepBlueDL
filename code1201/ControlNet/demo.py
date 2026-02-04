# -*- coding: utf-8 -*-
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from PIL import Image
import numpy as np
import cv2 as cv
from einops import einops

from annotator.canny import CannyDetector
from annotator.util import HWC3, resize_image
from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict


def t0():
    model = create_model('./models/cldm_v15.yaml').cpu()
    print(model)
    print(f"总参数量:{len(model.state_dict())}")
    missing_keys, unexpected_keys = model.load_state_dict(
        load_state_dict('./models/control_sd15_canny.pth', location='cpu'),
        strict=False
    )
    print(f"missing_keys: {missing_keys}")
    print(f"unexpected_keys: {unexpected_keys}")


def t1():
    eta = 0.0
    scale = 9.0
    num_samples = 2
    ddim_steps = 10
    low_threshold = 100
    high_threshold = 100
    image_resolution = 512
    prompt = "bird"
    a_prompt = "best quality, extremely detailed"
    n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    apply_canny = CannyDetector()  # 图像边缘检测算法 -> OpenCV的canny算法
    model = create_model('./models/cldm_v15.yaml').cpu()
    # model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cpu'), strict=True)
    model = model.cpu()
    ddim_sampler = DDIMSampler(model)

    input_image = Image.open("bird.jpg")
    input_image = np.asarray(input_image)
    print(input_image.shape)

    img = resize_image(HWC3(input_image), image_resolution)  # 转换成[H,W,C]结构，并reshape大小
    H, W, C = img.shape
    print(img.shape)

    detected_map = apply_canny(img, low_threshold, high_threshold)  # 得到controlnet要求输入的额外信息
    detected_map = HWC3(detected_map)
    print(detected_map.shape)

    cv.imshow('img', detected_map)
    cv.waitKey(0)
    cv.destroyAllWindows()

    control = torch.from_numpy(detected_map.copy()).float().cpu() / 255.0  # 入参数据
    control = torch.stack([control for _ in range(num_samples)], dim=0)  # [N,H,W,C]
    print(control.shape)
    control = einops.rearrange(control, 'n h w c -> n c h w').clone()  # 维度交换
    print(control.shape)

    cond = {
        "c_concat": [control],
        "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]
    }
    un_cond = {
        "c_concat": [control],
        "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]

    }
    shape = (4, H // 8, W // 8)

    model.control_scales = ([1] * 13)
    samples, intermediates = ddim_sampler.sample(
        ddim_steps, num_samples,
        shape, cond, verbose=False, eta=eta,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=un_cond
    )

    x_samples = model.decode_first_stage(samples)  # decode生成图像
    x_samples = einops.rearrange(x_samples, 'n c h w -> n h w c') * 127.5 + 127.5
    x_samples = x_samples.cpu().numpy().clip(0, 255).astype(np.uint8)

    for i in range(num_samples):
        cv.imshow(f'img:{i}', x_samples[i])
    cv.waitKey(0)
    cv.destroyAllWindows()




if __name__ == '__main__':
    t0()
