""" Parts of the U-Net model """
"""https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# noinspection PyTypeChecker
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: 上一层的输出(解码器过程中的)
        x2: 对应的encoder层的输出
        NOTE: 一般情况下,x1的feature map大小是x2的feature map大小的一半
        """
        x1 = self.up(x1)  # 上采样，扩大成原来的两倍
        _, _, x1h, x1w = x1.shape
        _, _, x2h, x2w = x2.shape
        if x1h != x2h or x1w != x2w:
            _py = torch.tensor([x2h - x1h])
            _px = torch.tensor([x2w - x1w])
            x1 = F.pad(x1, [_px // 2, _px - _px // 2, _py // 2, _py - _py // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# noinspection PyTypeChecker
class ELAN(nn.Module):
    def __init__(self, in_channel, out_channel=None):
        super(ELAN, self).__init__()
        if out_channel is None:
            assert in_channel % 2 == 0, "输入通道数目必须是2的倍数"
        else:
            assert out_channel % 4 == 0, "输入通道数目必须是4的倍数"

        ic = in_channel
        oc = (ic // 2) if out_channel is None else (out_channel // 4)
        self.part1 = nn.Sequential(
            nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=oc),
            nn.SiLU()
        )
        self.part2 = nn.Sequential(
            nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=oc),
            nn.SiLU()
        )
        self.part3 = nn.Sequential(
            nn.Conv2d(oc, oc, 3, 1, padding='same'),
            nn.BatchNorm2d(oc),
            nn.SiLU(),
            nn.Conv2d(oc, oc, 3, 1, padding='same'),
            nn.BatchNorm2d(oc),
            nn.SiLU()
        )
        self.part4 = nn.Sequential(
            nn.Conv2d(oc, oc, 3, 1, padding='same'),
            nn.BatchNorm2d(oc),
            nn.SiLU(),
            nn.Conv2d(oc, oc, 3, 1, padding='same'),
            nn.BatchNorm2d(oc),
            nn.SiLU()
        )
        self.part5 = nn.Sequential(
            nn.Conv2d(4 * oc, 4 * oc, 1, 1),
            nn.BatchNorm2d(4 * oc),
            nn.SiLU(),
        )

    def forward(self, x):
        x1 = self.part1(x)
        x2 = self.part2(x)
        x3 = self.part3(x2)
        x4 = self.part4(x3)
        x = torch.concat([x1, x2, x3, x4], dim=1)
        x = self.part5(x)
        return x


# noinspection PyTypeChecker
class MP(nn.Module):
    def __init__(self, ic):
        super(MP, self).__init__()
        oc = ic // 2
        self.part1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(ic, oc, 1, 1),
            nn.BatchNorm2d(oc),
            nn.SiLU(),
        )

        self.part2 = nn.Sequential(
            nn.Conv2d(ic, oc, 1, 1),
            nn.BatchNorm2d(oc),
            nn.SiLU(),
            nn.Conv2d(oc, oc, 3, 2, padding=1),
            nn.BatchNorm2d(oc),
            nn.SiLU()
        )

    def forward(self, x):
        x1 = self.part1(x)
        x2 = self.part2(x)
        return torch.concat([x1, x2], dim=1)


# noinspection PyTypeChecker
class UMP(nn.Module):
    def __init__(self, ic):
        super(UMP, self).__init__()
        oc = ic // 2
        self.part1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ic, oc, 1, 1),
            nn.BatchNorm2d(oc),
            nn.SiLU(),
        )

        self.part2 = nn.Sequential(
            nn.ConvTranspose2d(ic, oc, kernel_size=2, stride=2),
            nn.BatchNorm2d(oc),
            nn.SiLU(),
            nn.Conv2d(oc, oc, 3, 1, padding='same'),
            nn.BatchNorm2d(oc),
            nn.SiLU()
        )

    def forward(self, x):
        x1 = self.part1(x)
        x2 = self.part2(x)
        return torch.concat([x1, x2], dim=1)


class UNetUpWithELAN(nn.Module):
    def __init__(self, ic):
        super(UNetUpWithELAN, self).__init__()
        self.up = UMP(ic)
        self.elan = ELAN(ic * 2, ic // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.concat([x1, x2], dim=1)
        return self.elan(x)
