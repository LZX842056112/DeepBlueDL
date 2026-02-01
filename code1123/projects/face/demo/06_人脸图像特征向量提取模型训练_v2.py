# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/23 15:39
Create User : 19410
Desc : xxx
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from facenet_pytorch import InceptionResnetV1, training


class CrossEntropyLossWithTheta(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(CrossEntropyLossWithTheta, self).__init__()
        self.w = nn.Parameter(torch.randn(feature_dim, num_classes))  # 全连接输出对应的参数w
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pred, targets):
        """
        损失函数
        :param pred: [bs,e] 每个样本e维的特征向量
        :param targets: [bs] 每个样本实际类别id
        :return:
        """
        # ==== 普通交叉熵损失函数的写法
        # # 1. 获取每个样本属于各个类别的置信度
        # z = torch.matmul(pred, self.w)  # [bs,e] * [e,num_classes] --> [bs, num_classes]
        # # 2. 计算损失
        # loss = self.loss_fn(z, targets)

        # ==== 以夹角的方式计算损失
        pred = F.normalize(pred, p=2, dim=1)
        w = F.normalize(self.w, p=2, dim=0)
        cos_theta = torch.matmul(pred, w)  # 两个向量夹角θ的余弦值
        loss = self.loss_fn(cos_theta, targets)

        return loss

    def get_pred_scores(self, pred):
        pred = F.normalize(pred, p=2, dim=1)
        w = F.normalize(self.w, p=2, dim=0)
        cos_theta = torch.matmul(pred, w)  # 两个向量夹角θ的余弦值
        return cos_theta
        # return torch.matmul(pred, self.w)


def build_accuracy_fn(loss_fn):
    def accuracy(pred, y):
        logits = loss_fn.get_pred_scores(pred)
        _, preds = torch.max(logits, 1)
        return (preds == y).float().mean()

    return accuracy


def training_with_softmax_theta():
    # 训练数据所在文件夹，每个子文件就是一个人(也就是一个类别)
    data_dir = './datas/LFW/images_cropped'

    batch_size = 32
    epochs = 8
    workers = 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    resnet = InceptionResnetV1(
        classify=False,
        pretrained='vggface2'
    ).to(device)
    print(resnet)

    optimizer = optim.Adam(resnet.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, [5, 10])

    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(data_dir, transform=trans)
    img_inds = np.arange(len(dataset))
    np.random.shuffle(img_inds)
    train_inds = img_inds[:int(0.8 * len(img_inds))]
    val_inds = img_inds[int(0.8 * len(img_inds)):]

    train_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_inds)
    )
    val_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_inds)
    )

    loss_fn = CrossEntropyLossWithTheta(512, 6)  # 交叉熵损失函数
    metrics = {
        'fps': training.BatchTimer(),  # 效率指标
        'acc': build_accuracy_fn(loss_fn)  # 评估指标
    }

    writer = SummaryWriter(log_dir="./output/v2/summary")
    writer.iteration, writer.interval = 0, 10

    print('\n\nInitial')
    print('-' * 10)
    resnet.eval()
    training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        resnet.train()
        training.pass_epoch(
            resnet, loss_fn, train_loader, optimizer, scheduler,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

        resnet.eval()
        training.pass_epoch(
            resnet, loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

    writer.close()

    torch.save(resnet.state_dict(), './output/v2/model_param.pkl')


if __name__ == '__main__':
    training_with_softmax_theta()
