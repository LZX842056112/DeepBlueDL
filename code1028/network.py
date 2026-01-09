# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/28 20:13
Create User : 19410
Desc : 定义一些模型相关的代码
"""
import torch
import torch.nn as nn


class Conv2DBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if stride == 1 and padding is None:
            padding = (kernel_size - stride) // 2 + (1 if kernel_size % 2 == 0 else 0)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Inception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.b1 = Conv2DBNRelu(in_channels, out_channels[0][0], 1)
        self.b2 = nn.Sequential(
            Conv2DBNRelu(in_channels, out_channels[1][0], 1),
            Conv2DBNRelu(out_channels[1][0], out_channels[1][1])
        )
        self.b3 = nn.Sequential(
            Conv2DBNRelu(in_channels, out_channels[2][0], 1),
            Conv2DBNRelu(out_channels[2][0], out_channels[2][1], 5)
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, 1, padding=1),
            Conv2DBNRelu(in_channels, out_channels[3][0], 1)
        )

    def forward(self, x):
        x1 = self.b1(x)
        x2 = self.b2(x)
        x3 = self.b3(x)
        x4 = self.b4(x)
        return torch.concat([x1, x2, x3, x4], dim=1)


class SEModule(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        c = int(max(channels // r, 1))
        layers = [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, c, 1),
            nn.ReLU(),
            nn.Conv2d(c, channels, 1),
            nn.Sigmoid()
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        SE 模块
        :param x: [n,c,h,w]
        :return: [n,c,h,w]
        """
        alpha = self.layers(x)  # [n,c,1,1]
        return alpha * x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, has_se=False):
        super().__init__()
        if in_channels == out_channels:
            self.b1 = nn.Sequential(
                Conv2DBNRelu(in_channels, out_channels, 3),
                Conv2DBNRelu(out_channels, out_channels, 3)
            )
            self.b2 = nn.Identity()
        else:
            # 下采样
            self.b1 = nn.Sequential(
                Conv2DBNRelu(in_channels, out_channels, 3, 2, 1),
                Conv2DBNRelu(out_channels, out_channels, 3)
            )
            self.b2 = nn.Sequential(
                nn.MaxPool2d(3, 2, 1),
                Conv2DBNRelu(in_channels, out_channels, 1)
            )
        if has_se:
            self.se = SEModule(out_channels)
        else:
            self.se = nn.Identity()

    def forward(self, x):
        z1 = self.b1(x)
        z1 = self.se(z1)  # SE模块处理

        z2 = self.b2(x)

        return z1 + z2


class VGG(nn.Module):
    def __init__(self, num_classes, layers=None):
        super().__init__()
        if layers is None:
            layers = [
                64, 64, 'M',
                128, 128, 'M',
                256, 256, 256, 'M',
                512, 512, 512, 'M',
                512, 512, 512, 'M'
            ]

        # 解析逻辑
        in_channels = 3
        features = []
        for layer in layers:
            if isinstance(layer, int):
                m = Conv2DBNRelu(in_channels, layer)
                in_channels = layer  # 将当前层/当前模块的输出通道数目作为下一个模块的输入通道
            elif isinstance(layer, str) and layer == 'M':
                m = nn.MaxPool2d(2, 2)
            elif isinstance(layer, tuple):
                if layer[0] == 'C':
                    layer_args = [in_channels]
                    layer_args.extend(list(layer[1:]))
                    m = Conv2DBNRelu(*layer_args)
                    in_channels = layer_args[1]  # 将当前层/当前模块的输出通道数目作为下一个模块的输入通道
                else:
                    raise ValueError(f"参数异常: {layer}")
            else:
                raise ValueError(f"参数异常: {layer}")
            features.append(m)
        features.append(nn.AdaptiveAvgPool2d(7))
        self.features = nn.Sequential(*features)

        self.classify = nn.Sequential(
            nn.Linear(7 * 7 * in_channels, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        """
        前向过程
        :param x: [bs,c,h,w]
        :return:
        """
        z = self.features(x)
        z = torch.flatten(z, start_dim=1)  # [n,c,h,w] --> [n, c*h*w]
        z = self.classify(z)
        return z


class GoogLeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        features = [
            Conv2DBNRelu(3, 64, 7, 2, 3),
            nn.MaxPool2d(3, 2, 1),
            Conv2DBNRelu(64, 64, 1),
            Conv2DBNRelu(64, 192, 3),
            nn.MaxPool2d(3, 2, 1),
            Inception(192, [[64], [96, 128], [16, 32], [32]]),
            Inception(256, [[128], [128, 192], [32, 96], [64]]),
            nn.MaxPool2d(3, 2, 1),
            Inception(480, [[192], [96, 208], [16, 48], [64]]),
            Inception(512, [[160], [112, 224], [24, 64], [64]]),
            Inception(512, [[128], [128, 256], [24, 64], [64]]),
            Inception(512, [[112], [144, 288], [32, 64], [64]]),
            Inception(528, [[256], [160, 320], [32, 128], [128]]),
            nn.MaxPool2d(3, 2, 1),
            Inception(832, [[256], [160, 320], [32, 128], [128]]),
            Inception(832, [[384], [192, 384], [48, 128], [128]]),
            nn.AdaptiveAvgPool2d(1)
        ]
        self.features = nn.Sequential(*features)

        self.classify = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        """
        前向过程
        :param x: [bs,c,h,w]
        :return:
        """
        z = self.features(x)
        z = torch.flatten(z, start_dim=1)  # [n,c,h,w] --> [n, c*h*w]
        z = self.classify(z)
        return z


class ResNet(nn.Module):
    def __init__(self, num_classes, has_se=True):
        super().__init__()
        features = [
            Conv2DBNRelu(3, 64, 7, 2, 3),

            nn.MaxPool2d(2, 2),

            ResBlock(64, 64, has_se),
            ResBlock(64, 64, has_se),
            ResBlock(64, 64, has_se),

            ResBlock(64, 128),

            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),

            ResBlock(128, 256),

            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),

            ResBlock(256, 512),

            ResBlock(512, 512),
            ResBlock(512, 512),

            nn.AdaptiveAvgPool2d(1)
        ]

        self.features = nn.Sequential(*features)

        self.classify = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        前向过程
        :param x: [bs,c,h,w]
        :return:
        """
        z = self.features(x)
        z = torch.flatten(z, start_dim=1)  # [n,c,h,w] --> [n, c*h*w]
        z = self.classify(z)
        return z
