# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/14 20:28
Create User : 19410
Desc : 17种花的模型训练器
"""

from ..data_utils import build_dataloader
from ..looses import build_losses
from ..network import build_network
from ..optim_utils import build_optim, OptimArgs
from ..trainer import Trainer


class FlowersTrainer(Trainer):
    def __init__(self, train_path: str, val_path: str,
                 batch_size: int, total_epoch: int, dump_model_dir: str,
                 img_target_size=(100, 100)):
        # 1. 数据加载 数据处理
        _, train_dataloader, class_names = build_dataloader(
            data_dir=train_path, batch_size=batch_size, shuffle=True,
            target_size=img_target_size
        )
        _, val_dataloader, _ = build_dataloader(
            data_dir=val_path, batch_size=batch_size * 2, shuffle=False,
            target_size=img_target_size,
            class_names=class_names
        )

        net = build_network(
            num_classes=len(class_names),
            in_channels=3
        )
        loss_fn = build_losses()
        train_opt = build_optim(
            args=OptimArgs(
                net=net,
                lr=0.01
            )
        )
        super().__init__(
            net=net,
            loss_fn=loss_fn,
            train_opt=train_opt,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            total_epoch=total_epoch,
            dump_model_dir=dump_model_dir,
            class_names=class_names
        )
