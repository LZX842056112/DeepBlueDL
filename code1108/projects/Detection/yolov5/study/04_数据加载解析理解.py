# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/2 16:45
Create User : 19410
Desc : xxx
"""
from utils.dataloaders import create_dataloader
from utils.general import colorstr


def t0():
    train_path = r'D:\\Projects\\DeepBule\\d2l-zh\\code1108\\projects\\Detection\\yolov5\\data\\coco128\\images\\train2017'
    imgsz = 640
    batch_size = 4
    WORLD_SIZE = 1
    gs = 32
    single_cls = False
    hyp = {
        'anchor_t': 4.0, 'box': 0.05, 'cls': 0.5, 'cls_pw': 1.0, 'copy_paste': 0.0, 'degrees': 0.0, 'fl_gamma': 0.0,
        'fliplr': 0.5, 'flipud': 0.0, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'iou_t': 0.2, 'lr0': 0.01,
        'lrf': 0.01, 'mixup': 0.0, 'momentum': 0.937, 'mosaic': 1.0, 'obj': 1.0, 'obj_pw': 1.0, 'perspective': 0.0,
        'scale': 0.5, 'shear': 0.0, 'translate': 0.1, 'warmup_bias_lr': 0.1, 'warmup_epochs': 3.0,
        'warmup_momentum': 0.8, 'weight_decay': 0.0005
    }

    train_loader, dataset = create_dataloader(
        train_path,  # 图像所在的文件夹路径，也就是在data.yaml中配置的train对应的绝对路径
        imgsz,  # 给定DataLoader返回的图像尺度大小
        batch_size // WORLD_SIZE,  # 批次大小
        gs,  # 网络的Feature Map缩放系数/倍数
        single_cls,  # 是否是单类别的数据加载，如果为True，那么强制将返回的cls_id设置为0
        hyp=hyp,  # 超参数
        augment=True,  # 是否做数据增强
        cache=None,  # 是否提前加载img图像数据
        rect=False,
        rank=-1,
        workers=0,  # 数据加载的多进程数量
        image_weights=False,  # 是否运行过程中针对图像类别加权
        quad=False,
        prefix=colorstr('train: '),
        shuffle=True  # 是否打乱数据的顺序
    )

    for batch in train_loader:
        print(type(batch))
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[2])
        break


if __name__ == '__main__':
    t0()
