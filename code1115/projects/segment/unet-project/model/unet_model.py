""" Full assembly of the parts to form the complete network """
"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)  # 连续的两次卷积, [N,64,512,512]
        x2 = self.down1(x1)  # [N,128,256,256]
        x3 = self.down2(x2)  # [N,256,128,128]
        x4 = self.down3(x3)  # [N,512,64,64]
        x5 = self.down4(x4)  # [N,512,32,32]
        x = self.up1(x5, x4)  # 输入参数是：上一层的输出、对应的encoder层的输出
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNetWithELAN(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetWithELAN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = nn.Sequential(
            MP(128),
            ELAN(128)
        )
        self.down3 = nn.Sequential(
            MP(256),
            ELAN(256)
        )
        self.down4 = nn.Sequential(
            MP(512),
            ELAN(512, 512)  # 输入512通道，输出512通道
        )

        self.up1 = UNetUpWithELAN(512)  # 输入512*2 --> 256
        self.up2 = UNetUpWithELAN(256)  # 输入256*2 --> 128
        self.up3 = UNetUpWithELAN(128)  # 输入128*2 --> 64
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)  # 连续的两次卷积, [N,64,512,512]
        x2 = self.down1(x1)  # [N,128,256,256]
        x3 = self.down2(x2)  # [N,256,128,128]
        x4 = self.down3(x3)  # [N,512,64,64]
        x5 = self.down4(x4)  # [N,512,32,32]
        x = self.up1(x5, x4)  # 输入参数是：上一层的输出、对应的encoder层的输出
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def fuse(self):
        """
        最好实现一下算子的融合(在线推理部署预测的时候)
        """
        pass
