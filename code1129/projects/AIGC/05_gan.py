# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/29 17:39
Create User : 19410
Desc : GAN
"""

import torch
from torch import nn
import torch.optim as optim


class DNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.discriminator_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),  # [N,1,28,28] -> [N,16,28,28]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # [N,16,28,28] -> [N,16,14,14]
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),  # [N,16,14,14] -> [N,32,14,14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # [N,16,28,28] -> [N,32,7,7]
        )
        self.discriminator_linear = nn.Sequential(
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def discriminator(self, x):
        x = self.discriminator_conv(x)  # [N,1,28,28] -> [N,32,7,7]
        b, c, h, w = x.shape
        x = x.view(b, -1)  # [N,32,7,7] -> [N,32*7*7]
        x = self.discriminator_linear(x)  # [N,32*7*7] -> [N,1]
        return x[:, 0]  # N个样本，每个样本输入真实类别的置信度

    def forward(self, img):
        return self.discriminator(img)


class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()

        self.decoder_linear = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 32 * 7 * 7),
            nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),  # [N,32,7,7] -> [N,32,14,14]
            nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),  # [N,32,14,14] -> [N,16,14,14]
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # [N,16,14,14] -> [N,16,28,28]
            nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=1),  # [N,16,28,28] -> [N,1,28,28]
        )

        # 判别器网络
        self.d = DNetwork()

        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def generator(self, x):
        x = self.decoder_linear(x)  # [N,10] -> [N,32*7*7]
        x = x.view(x.shape[0], 32, 7, 7)
        x = self.decoder_conv(x)  # [N,32,7,7] -> [N,1,28,28]
        x = x.sigmoid()  # [N,1,28,28] 取值范围变成了0~1之间
        x = 2 * x - 0.5  # 取值范围就变成了 -0.5~1.5 之间
        x = torch.clip(x, 0.0, 1.0)
        return x  # [N,1,28,28]

    def forward(self, nosie, rel_img):
        n1 = nosie.shape[0]
        n2 = rel_img.shape[0]

        # 1. 生成网络 输出构建的图像
        pred_img = self.generator(nosie)  # [N1,10] -> [N1,1,28,28]

        # 2. 生成器的损失 --> 生成器期望生成的数据能够以假乱真 --> 也就是希望判别器针对生成图像这部分数据预测为真实的置信度越高越好
        for param in self.d.parameters():
            param.requires_grad = False
        pred_score = self.d(pred_img)  # [N1,1,28,28] -> [N1] --> 生成器期望判别网络无法正确判断这些生成的样本
        loss_g = self.loss_fn(pred_score, torch.ones_like(pred_score))
        loss_g = loss_g.sum() / n1
        for param in self.d.parameters():
            param.requires_grad = True

        # 2. 判别器的损失 --> 分类损失: 反向传播应该只对判别器部分参数做更新
        rel_score = self.d(rel_img)  # [N2,1,28,28] -> [N2] --> 对应的标签就是1
        rel_loss_d = self.loss_fn(rel_score, torch.ones_like(rel_score))  # 如果图像判断为真实的预测置信度越高，loss就越小
        pred_score = self.d(pred_img.detach())  # [N1,1,28,28] -> [N1] --> 对应的标签就是0
        pred_loss_d = self.loss_fn(pred_score, torch.zeros_like(pred_score))  # 如果图像判断为虚假的预测置信度越高，loss就越小
        loss_d = (rel_loss_d.sum() + pred_loss_d.sum()) / (n1 + n2)

        return loss_d, loss_g


def t0():
    net = GAN()
    opt1 = optim.SGD(params=[*net.decoder_conv.parameters(), *net.decoder_linear.parameters()], lr=0.001)
    opt2 = optim.SGD(params=[*net.d.parameters()], lr=0.001)

    loss_d, loss_g = net(
        torch.randn(4, 10),  # 随机4个样本，每个样本10维的特征向量
        torch.rand(8, 1, 28, 28)  # 模拟真实图像的输入，表示8个真实图像的输入
    )
    print(loss_g)
    print(loss_d)

    print("\n生成器的反向传播:")
    print(f"生成器部分参数梯度:{net.decoder_linear[-2].bias.grad}")
    print(f"判别器部分参数梯度:{net.d.discriminator_linear[-1].bias.grad}")
    loss_g.backward(retain_graph=True)  # 反向传播，求解梯度，仅对判别部分的参数进行了梯度的计算
    opt2.zero_grad()  # loss_g在求解梯度时候，会包含生成部分、判别部分两部分参数的梯度，所以针对判别部分的梯度重置为0
    print(f"生成器部分参数梯度:{net.decoder_linear[-2].bias.grad}")
    print(f"判别器部分参数梯度:{net.d.discriminator_linear[-1].bias.grad}")
    opt1.step()
    opt1.zero_grad()

    print("\n判别器的反向传播:")
    print(f"生成器部分参数梯度:{net.decoder_linear[-2].bias.grad}")
    print(f"判别器部分参数梯度:{net.d.discriminator_linear[-1].bias.grad}")
    loss_d.backward()  # 反向传播，求解梯度，仅对判别部分的参数进行了梯度的计算
    print(f"生成器部分参数梯度:{net.decoder_linear[-2].bias.grad}")
    print(f"判别器部分参数梯度:{net.d.discriminator_linear[-1].bias.grad}")
    opt2.step()
    opt2.zero_grad()


if __name__ == '__main__':
    t0()
