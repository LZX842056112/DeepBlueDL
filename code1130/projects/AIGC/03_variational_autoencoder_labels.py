# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/29 16:40
Create User : 19410
Desc : VAE + Labels
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image


class ConvBnRelu(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, (3, 3), stride=(1, 1), padding=(1, 1))
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU()

    def forward(self, img):
        return self.act(self.bn(self.conv(img)))


class UPSampleConvBnRelu(ConvBnRelu):
    def __init__(self, in_channel, out_channel):
        super(UPSampleConvBnRelu, self).__init__(in_channel, out_channel)
        # (7-1)*2 - 2 * 1 + 1*(3-1) + 1 + 1
        # (14-1)*2 - 2 * 1 + 1*(3-1) + 1 + 1
        self.conv = nn.ConvTranspose2d(
            in_channel, out_channel, (3, 3), stride=(2, 2), padding=(1, 1),
            output_padding=(1, 1)
        )


class MNISTAutoEncoder(nn.Module):
    def __init__(self):
        super(MNISTAutoEncoder, self).__init__()
        self.label_emb = nn.Embedding(10, 128)  # 内部实际上就是一个矩阵，大小为[10,128] 也就是每个数字对应一个128维的向量
        self.encoder = nn.Sequential(
            ConvBnRelu(1, 16),
            nn.MaxPool2d(2, 2),  # 14*14
            ConvBnRelu(16, 32),
            ConvBnRelu(32, 32),
            nn.MaxPool2d(2, 2),  # 7*7
            ConvBnRelu(32, 64),
            ConvBnRelu(64, 64),
            nn.AdaptiveMaxPool2d(7)
        )

        self.fc_mu = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128)
        )
        self.fc_var = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128)
        )

        # NOTE: 解码器除了随机的向量作为输入外，可以额外增加一个类别向量 --> 目的：为了看是否能够通过给定label来生成对应的数据
        self.fc_decoder = nn.Sequential(
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 64 * 7 * 7)
        )
        self.decoder = nn.Sequential(
            ConvBnRelu(64, 32),
            UPSampleConvBnRelu(32, 32),
            ConvBnRelu(32, 32),
            ConvBnRelu(32, 16),
            UPSampleConvBnRelu(16, 16),
            ConvBnRelu(16, 1)
        )

    def _encode(self, img):
        z = self.encoder(img)  # [bs,1,28,28] -> [bs,64,7,7]
        z = torch.flatten(z, start_dim=1)  # [bs,64,7,7] -> [bs, 64*7*7]
        mu = self.fc_mu(z)  # [bs,128]
        log_var = self.fc_var(z)  # [bs,128]
        return mu, log_var

    def _decode(self, z):
        z = self.fc_decoder(z)
        z = z.view(-1, 64, 7, 7)
        z = self.decoder(z)
        # z = torch.tanh(z) + 0.5
        return z

    def _reparameterize(self, mu, logvar, labels):
        std = torch.exp(0.5 * logvar)  # 得到标准差
        eps = torch.randn_like(std)  # 随机一个数据(一个标准的正太分布 均值为0 方差为1)
        rnd_z = eps * std + mu  # 满足均值为mu，标准差为std的随机数据
        label_z = self.label_emb(labels)  # 获取每个label对应的128维的特征向量
        return rnd_z + label_z

    def forward(self, img, labels):
        mu, log_var = self._encode(img)  # 编码器过程，得到数据对应数据分布的均值和方差

        decoder_input = self._reparameterize(mu, log_var, labels)  # 随机一个满足数据分布的特征向量

        z = self._decode(decoder_input)  # 解码器解码
        return z, mu, log_var

    def rand_image(self, img_num, device, labels=None, use_labels=True, only_labels=False):
        decoder_input = torch.randn(img_num, 128, device=device)  # 随机一个标准正太分布的的向量

        if use_labels:
            if labels is None:
                labels = torch.randint(low=0, high=10, size=(img_num,), device=device)
            label_z = self.label_emb(labels)
            if only_labels:
                decoder_input = label_z
            else:
                decoder_input = decoder_input + label_z

        samples = self._decode(decoder_input)
        return samples


@torch.no_grad()
def do_eval(epoch, net: MNISTAutoEncoder, test_dataloader, loss_fn, output_img_dir, device):
    net.eval()
    for test_image, test_labels in test_dataloader:
        test_image = test_image.to(device)
        test_labels = test_labels.to(device)
        pred_image, mu, log_var = net(test_image, test_labels)

        recons_loss = loss_fn(pred_image, test_image)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        test_loss = recons_loss + 0.001 * kld_loss

        print(f"Epoch:{epoch} Eval Loss {recons_loss.item():.3f} + {kld_loss.item():.3f} = {test_loss.item():.3f}")

        # 随机生成一张图像
        rand_image1 = net.rand_image(
            img_num=test_image.shape[0],
            device=device,
            labels=test_labels  # 给定标签的随机图像
        )
        rand_image2 = net.rand_image(
            img_num=test_image.shape[0],
            device=device
        )

        # 保存图像
        test_img = make_grid(test_image, padding=5)  # 原始图像
        pred_img = make_grid(torch.clip(pred_image, 0.0, 1.0), padding=5)  # 原始图像模型 -> 输出图像
        rand_image1 = make_grid(torch.clip(rand_image1, 0.0, 1.0), padding=5)  # 原始标签输入模型 -> 输出图像
        rand_image2 = make_grid(torch.clip(rand_image2, 0.0, 1.0), padding=5)  # 随机标签输入模型 -> 输出图像
        grid = make_grid(torch.stack([test_img, pred_img, rand_image1, rand_image2], dim=0), nrow=2, padding=10,
                         pad_value=0.5)

        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        im.save(os.path.join(output_img_dir, f"{epoch:04d}.png"))

        # 每个epoch仅随机一个batch的数据进行check
        break


def do_train(epoch, net, train_dataloader, loss_fn, train_op, global_train_step, device):
    train_step = global_train_step
    net.train()
    for image, labels in train_dataloader:
        image = image.to(device)
        labels = labels.to(device)

        # 前向处理
        pred_image, mu, log_var = net(image, labels)
        recons_loss = loss_fn(pred_image, image)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + 0.001 * kld_loss

        # 反向处理
        train_op.zero_grad()
        loss.backward()
        train_op.step()
        if train_step % 100 == 0:
            # print(f"mu: {mu}")
            # print(f"log var:{log_var}")
            print(f"Epoch:{epoch} Train Step:{train_step} "
                  f"Train Loss {recons_loss.item():.3f} + {kld_loss.item():.3f} = {loss.item():.3f}")
        train_step += 1
    return train_step


def do_save(epoch, net: MNISTAutoEncoder, output_save_dir):
    os.makedirs(output_save_dir, exist_ok=True)
    obj = {
        'epoch': epoch,
        'net': net.state_dict()
    }
    torch.save(obj, os.path.join(output_save_dir, f"{epoch:04d}.pkl"))


def training():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ts = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(28)
    ])
    train_dataset = datasets.MNIST(
        root='../../datas/MNIST',  # 数据保存的文件夹路径
        train=True,
        transform=ts,
        download=True  # 如果在给定的文件夹中不存在数据，那么进行数据下载
    )
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, num_workers=0)
    test_dataset = datasets.MNIST(
        root='../../datas/MNIST',  # 数据保存的文件夹路径
        train=False,
        transform=ts,
        download=True  # 如果在给定的文件夹中不存在数据，那么进行数据下载
    )
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=32, num_workers=0)

    net = MNISTAutoEncoder()
    net.to(device=device)
    loss_fn = nn.SmoothL1Loss()
    train_op = optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.0)

    output_img_dir = "./output/03_vae_labels/eval_image/"
    output_save_dir = "./output/03_vae_labels/save_model/"
    os.makedirs(output_img_dir, exist_ok=True)
    train_step = 0

    do_eval(0, net, test_dataloader, loss_fn, output_img_dir, device)
    for epoch in range(1, 100):
        # 训练
        train_step = do_train(epoch, net, train_dataloader, loss_fn, train_op, train_step, device)

        # 测试
        do_eval(epoch, net, test_dataloader, loss_fn, output_img_dir, device)

        # 模型保存
        do_save(epoch, net, output_save_dir)


@torch.no_grad()
def random_image():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = MNISTAutoEncoder()
    state = torch.load("./output/03_vae_labels/save_model/0004.pkl", map_location='cpu')
    net.load_state_dict(state['net'])
    net = net.to(device).eval()

    labels = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 0, 0], device=device)
    bs = labels.shape[0]

    # 基于完全随机向量的生成
    rand_image1 = net.rand_image(bs, device, labels=labels, use_labels=False, only_labels=False)
    # 基于随机向量 + 标签向量的生成
    rand_image2 = net.rand_image(bs, device, labels=labels, use_labels=True, only_labels=False)
    # 只使用标签向量的生成
    rand_image3 = net.rand_image(bs, device, labels=labels, use_labels=True, only_labels=True)

    def _save_img(_img, _path):
        _img = make_grid(torch.clip(_img, 0.0, 1.0), nrow=2, padding=5, pad_value=0.5)
        ndarr = _img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        w, h = im.size
        im = im.resize(size=(w * 4, h * 4))
        im.save(_path)

    _save_img(rand_image1, "./output/03_vae_labels/rand_image1.png")
    _save_img(rand_image2, "./output/03_vae_labels/rand_image2.png")
    _save_img(rand_image3, "./output/03_vae_labels/rand_image3.png")


if __name__ == '__main__':
    # training()
    random_image()
