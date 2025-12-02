import torch
import torch.nn as nn
import cv2 as cv


def tt00():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1.加载图像
    img = cv.imread("../day1/xiaoren.png")

    # 2.将numpy转换为tensor对象
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.to(device)
    print(img_tensor.shape, img_tensor.dtype, img_tensor.device)

    # 维度的交换，当前下列代码的含义:将原来的第2维作为现在新数据中的第0维
    img_tensor = torch.permute(img_tensor, dims=(2, 0, 1))  # [C,H,W]
    img_tensor = img_tensor[None]

    # 3.卷积操作
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

    # 4.激活操作
    act = nn.PReLU()
    act = act.to(device=device)
    act_tensor = act(conv_tensor)
    print(act_tensor.shape)


def tt01():
    # act = nn.ReLU()
    act = nn.PReLU()

    x = torch.randn(4, 3, 30, 30)
    o = act(x)
    print(o.shape)

    print("=" * 30)
    # conv.named_parameters():torch中的模块通用方法，获取当前模块内部对应的所有参数(包含子模块内部的参数)
    for name, param in act.named_parameters():
        print(f"参数:{name} -- {param.shape} -- {param.requires_grad}")


def tt02():
    act = nn.ReLU()

    x = torch.randint(-5, 5, (1, 2, 6, 6))
    o = act(x)
    print(o.shape)
    print(x)
    print(o)

    print("=" * 30)
    # conv.named_parameters():torch中的模块通用方法，获取当前模块内部对应的所有参数(包含子模块内部的参数)
    for name, param in act.named_parameters():
        print(f"参数:{name} -- {param.shape} -- {param.requires_grad}")


if __name__ == "__main__":
    # tt01()
    # tt02()
    tt00()
