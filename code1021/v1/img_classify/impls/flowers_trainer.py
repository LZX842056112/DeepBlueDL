# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/14 20:28
Create User : 19410
Desc : 17种花的模型训练器
"""
import os.path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils

from ..data_utils import build_dataloader, DataAugmentArgs
from ..looses import build_losses
from ..network import build_network
from ..optim_utils import build_optim, OptimArgs
from ..trainer import Trainer


class FlowersTrainer(Trainer):
    def __init__(self, train_path: str, val_path: str,
                 batch_size: int, total_epoch: int, root_dir: str,
                 img_target_size=(100, 100)):
        # 1. 数据加载 数据处理
        _, train_dataloader, class_names = build_dataloader(
            data_dir=train_path, batch_size=batch_size, shuffle=True,
            target_size=img_target_size,
            is_augment=True,
            aug_args=DataAugmentArgs(
                p_rota=0.8
            )
        )
        _, val_dataloader, _ = build_dataloader(
            data_dir=val_path, batch_size=batch_size * 2, shuffle=False,
            target_size=img_target_size,
            class_names=class_names,
            is_augment=False,
        )

        net = build_network(
            num_classes=len(class_names),
            in_channels=3
        )
        print(net)
        loss_fn = build_losses()
        train_opt = build_optim(
            args=OptimArgs(
                net=net,
                lr=0.01
            )
        )

        dump_model_dir = os.path.join(root_dir, "models")
        summary_log_dir = os.path.join(root_dir, "logs")
        super().__init__(
            net=net,
            loss_fn=loss_fn,
            train_opt=train_opt,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            total_epoch=total_epoch,
            dump_model_dir=dump_model_dir,
            class_names=class_names,
            summary_writer=self.build_summary_writer(summary_log_dir, net)
        )

    def build_summary_writer(self, summary_log_dir, net):
        # PS: 如果环境中默认有tensorflow的环境的，那么默认应该会自带tensorboard，可以不安装下列版本，直接使用默认的
        # 安装tensorboard库  pip install tensorboard==2.13.0
        # 如果运行过程中，出现缺少包或者其他依赖库相关异常，那么直接进行新的安装即可: pip install six==1.16.0 protobuf==3.20.0
        # pip install six==1.16.0 protobuf==3.20.0 tensorboard==2.13.0
        # 运行完成后/运行过程中，在命令行，输入以下命令:
        # tensorboard --logdir xxx
        # tensorboard --logdir D:\Projects\DeepBule\d2l-zh\code1021\v1\output\logs
        writer = SummaryWriter(log_dir=summary_log_dir)

        def build_hook_func(that, module_name):
            def hook_func(m, m_args, m_outputs):
                tag = f"{'train' if m.training else 'eval'}_{module_name}"
                _steps = {}
                if hasattr(that, 'summary_steps'):
                    _steps = getattr(that, 'summary_steps')
                step = _steps.get(tag) or 1
                _steps[tag] = step + 1
                if step % 100 != 0:
                    return

                show_imgs = m_outputs[0:1]  # 仅对当前批次的第一张图像进行可视化 [1,C,H,W]
                show_imgs = torch.transpose(show_imgs, 0, 1)  # [C,1,H,W] 将每个通道看成一个图像
                # 多个通道合并成一个图像
                grid = utils.make_grid(show_imgs, pad_value=0.5, normalize=True)
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().to("cpu", torch.uint8).numpy()

                # 图像可视化输出
                writer.add_image(
                    tag, ndarr, global_step=step, dataformats='HWC'
                )

                return None

            return hook_func

        # 注册hooks
        hook_handles = []
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                _hook_fn = build_hook_func(
                    that=self,
                    module_name=f"{name}_conv"
                )
                handle = module.register_forward_hook(hook=_hook_fn)
                hook_handles.append(handle)
            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d)):
                _hook_fn = build_hook_func(
                    that=self,
                    module_name=f"{name}_pool"
                )
                handle = module.register_forward_hook(hook=_hook_fn)
                hook_handles.append(handle)
            elif isinstance(module, nn.BatchNorm2d):
                _hook_fn = build_hook_func(
                    that=self,
                    module_name=f"{name}_bn"
                )
                handle = module.register_forward_hook(hook=_hook_fn)
                hook_handles.append(handle)

        # 在writer中增加一个属性
        writer.hook_handles = hook_handles

        return writer
