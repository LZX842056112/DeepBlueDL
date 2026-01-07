# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/14 21:35
Create User : 19410
Desc : xxx

https://pai.console.aliyun.com/
"""
import sys

print(sys.path)
# 下列from...import...能够正常运行的前提条件是，classify_17flowers所在的父文件夹路径在sys.path环境变量中
from img_classify.impls.flowers_trainer import FlowersTrainer

if __name__ == '__main__':
    trainer = FlowersTrainer(
        train_path=r"../../code0923/17flowers/train",
        val_path=r"D../../code0923/17flowers/val",
        batch_size=8,
        total_epoch=100,
        root_dir="./output",
        img_target_size=(100, 100)
    )
    trainer.train()
