import os

# 设置环境变量 XDG_CACHE_HOME，指定缓存目录路径
os.environ['XDG_CACHE_HOME'] = r"D:\cv1201"

# 导入项目共享模块和配置文件
from share import *
import config

# 导入必要的库
import cv2  # OpenCV 库，用于图像处理
import einops  # 用于张量操作的工具库
import gradio as gr  # Gradio 库，用于构建 Web 界面
import numpy as np  # NumPy 库，用于数值计算
import torch  # PyTorch 深度学习框架
import random  # 随机数生成器

# 导入其他辅助模块
from pytorch_lightning import seed_everything  # 设置随机种子以保证可重复性
from annotator.util import resize_image, HWC3  # 图像缩放和格式转换工具
from annotator.canny import CannyDetector  # Canny 边缘检测器
from cldm.model import create_model, load_state_dict  # 控制扩散模型相关函数
from cldm.ddim_hacked import DDIMSampler  # DDIM 采样器

# 初始化 Canny 边缘检测器
apply_canny = CannyDetector()

# 创建并加载控制扩散模型
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(
    load_state_dict('./models/control_sd15_canny.pth', location='cpu'),
    strict=False
)
model = model.cpu()
ddim_sampler = DDIMSampler(model)  # 初始化 DDIM 采样器


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength,
            scale, seed, eta, low_threshold, high_threshold):
    """
    处理输入图像并生成结果图像的核心函数。

    参数说明：
    - input_image: 输入图像（NumPy 数组）
    - prompt: 正向提示词
    - a_prompt: 添加的正向提示词
    - n_prompt: 负向提示词
    - num_samples: 生成图像的数量
    - image_resolution: 输出图像分辨率
    - ddim_steps: DDIM 采样步数
    - guess_mode: 是否启用猜测模式
    - strength: 控制强度
    - scale: 引导尺度
    - seed: 随机种子
    - eta: DDIM 参数
    - low_threshold: Canny 低阈值
    - high_threshold: Canny 高阈值

    返回值：
    - 包含边缘图和生成图像的结果列表
    """
    print("开始处理")
    with torch.no_grad():  # 禁用梯度计算以节省内存
        # 缩放输入图像到目标分辨率
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape  # 获取图像的高度、宽度和通道数
        print(f"图像大小:{H} {W} {C}")

        # 使用 Canny 算法提取边缘图
        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        # 将边缘图转换为张量，并归一化到 [0, 1]
        control = torch.from_numpy(detected_map.copy()).float().cpu() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()  # 调整维度顺序

        # 设置随机种子
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        # 如果启用了内存优化，则切换模型状态
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        # 构造条件输入和无条件输入
        cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]
        }

        # 定义潜在空间形状
        shape = (4, H // 8, W // 8)

        # 切换模型状态为扩散模式
        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        # 设置控制尺度
        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
                [strength] * 13)

        # 执行 DDIM 采样
        samples, intermediates = ddim_sampler.sample(
            ddim_steps, num_samples,
            shape, cond, verbose=False, eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond
        )

        # 切换回非扩散模式
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        # 解码生成的潜在表示为图像
        x_samples = model.decode_first_stage(samples)
        x_samples = (
                einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5
        ).cpu().numpy().clip(0, 255).astype(np.uint8)  # 转换为 [0, 255] 的 RGB 图像

        # 提取生成图像
        results = [x_samples[i] for i in range(num_samples)]

    # 返回边缘图（反转颜色）和生成图像
    return [255 - detected_map] + results


# 构建 Gradio Web 页面
block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Canny Edge Maps")  # 页面标题
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")  # 图像上传组件
            prompt = gr.Textbox(label="Prompt")  # 正向提示词输入框
            run_button = gr.Button(label="Run")  # 运行按钮
            with gr.Accordion("Advanced options", open=False):  # 高级选项折叠面板
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1)
                high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2,
                                                                                                   height='auto')  # 结果展示区域

    # 绑定运行按钮与处理函数
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength,
           scale, seed, eta, low_threshold, high_threshold]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])

# 启动 Web 服务
block.launch(server_name='0.0.0.0')