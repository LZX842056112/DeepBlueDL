# -*- coding: utf-8 -*-
import cv2
import torch
from torchvision import utils
from torchvision.models import segmentation


def t0():
    model = segmentation.deeplabv3_mobilenet_v3_large(
        weights=segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    )
    print(model)

    img = torch.rand(2, 3, 256, 256)  # [N,3,img_h,img_w]
    result = model(img)
    print(type(result))
    out = result['out']  # 主分支的输出 [N,num_class,img_h,img_w] 每个图像像素属于各个类别的置信度
    aux = result.get('aux')  # 辅助分支的输出，可能为None  [N,num_class,img_h,img_w] 每个图像像素属于各个类别的置信度
    print(out.shape)
    print(aux.shape)


def t1():
    model = segmentation.deeplabv3_mobilenet_v3_large(
        weights=segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    )
    model.eval()
    print(model)

    img_file = "./dog.png"
    img_file = "./cat.jpg"
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # [H,W,3]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float32)[None]  # [1,3,H,W]
    result = model(img)
    print(type(result))
    out = result['out']  # 主分支的输出 [N,num_class,img_h,img_w] 每个图像像素属于各个类别的置信度
    print(out.shape)
    out = torch.sigmoid(out)  # 转换为概率 [1,21,H,W]
    out =  (out > 0.5).to(dtype=torch.float32)
    utils.save_image(
        tensor=torch.transpose(out, 0, 1),  # [1,21,H,W] -> [21,1H,W]
        fp=f"deeplabv3.png"
    )


if __name__ == '__main__':
    t1()
