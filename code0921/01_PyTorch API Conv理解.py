import cv2 as cv
import torch
import torch.nn as nn


def tt01():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1.加载图像
    # img:shape形状为[H,W,C，numpy格式 shape形状还是[H,W,C]
    img = cv.imread("../day1/xiaoren.png")

    # 2.将numpy转换为tensor对象
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor / 255.0
    # img_tensor = img_tensor.to(torch.float64)
    img_tensor = img_tensor.to(device)
    print(img_tensor.shape, img_tensor.dtype, img_tensor.device)

    # 维度的交换，当前下列代码的含义:将原来的第2维作为现在新数据中的第0维
    # 2 -->0
    # 0-->1
    # 1--> 2
    img_tensor = torch.permute(img_tensor, dims=(2, 0, 1))  # [C,H,W]
    # print(torch.max(torch.abs(img_tensor[:, :, 0] - img_tensor[0])))
    img_tensor = img_tensor[None]
    # torch.unsqueeze(img_tensor, dim=0)

    # 3.卷积操作
    '''
    torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
    dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
    '''
    conv = nn.Conv2d(
        in_channels=3,  # 当前卷积操作对应的输入通道数目，由上一层的输出决定
        out_channels=10,  # 当前卷积操作对应的输出通道数目，可自行任感决定
        kernel_size=3,  # 给定卷积核的大小
        stride=1,  # 给定卷积的步长大小
        padding=0  # 给定卷积过程中的填充大小
    )
    conv = conv.to(device=device)
    conv_tensor = conv(img_tensor)  # [N,C_in,H,W]-> [N,C_out,H,W]
    print(conv_tensor.shape)


def tt02():
    conv = nn.Conv2d(3, 20, 3, 1, 0)
    x = torch.randn(4, 3, 30, 30)
    o = conv(x)
    print(o.shape)

    print("=" * 30)
    # conv.named_parameters():torch中的模块通用方法，获取当前模块内部对应的所有参数(包含子模块内部的参数)
    for name, param in conv.named_parameters():
        print(f"参数:{name} -- {param.shape} -- {param.requires_grad}")


if __name__ == "__main__":
    # tt01()
    tt02()
