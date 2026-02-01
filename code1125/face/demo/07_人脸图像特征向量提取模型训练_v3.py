# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/23 15:39
Create User : 19410
Desc : xxx
"""
import math

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


class ArcFaceCrossEntropyLossWithTheta(nn.Module):
    """
    考虑用ArcFace损失
    """

    def __init__(self, feature_dim, num_classes, m=0.5, s=30):
        super(ArcFaceCrossEntropyLossWithTheta, self).__init__()
        self.w = nn.Parameter(torch.randn(feature_dim, num_classes))
        self.loss_fn = nn.CrossEntropyLoss()
        self.m = m
        self.s = s

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

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

        # # ==== 以夹角的方式计算损失 --> 普通损失
        # pred = F.normalize(pred, p=2, dim=1)
        # w = F.normalize(self.w, p=2, dim=0)
        # cos_theta = torch.matmul(pred, w)  # 两个向量夹角θ的余弦值
        # loss = self.loss_fn(cos_theta, targets)

        # 1. 对特征向量和参数w分别做L2 norm处理
        pred = F.normalize(pred, p=2, dim=1)  # 是为了保证 ||pred||==1
        w = F.normalize(self.w, p=2, dim=0)  # 是为了保证 ||w||==1
        cosine = torch.matmul(pred, w)  # 全连接 实际上就是cos(theta)的值

        # 2. 惩罚theta
        # 方式一：直接以反三角函数计算
        theta = torch.arccos(cosine)  # 通过反三角函数获得得到每个类别的夹角theta值
        phi = torch.cos(torch.clip(theta + self.m, 0.0, torch.pi))  # 以theta + m作为角度，计算置信度值

        # # 方式二：数学公式
        # sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1.0))
        # phi = cosine * self.cos_m - sine * self.sin_m
        # phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 3. 合并两部分的置信度：实际类别使用phi、其它使用cosine
        onehot = F.one_hot(targets.long(), cosine.shape[1]).to(phi.dtype)  # 实际标签位置为1，其它位置为0
        logits = cosine * (1.0 - onehot) + phi * onehot  # 实际标签位置使用cos(theta + m)，其它位置使用 cos(theta)
        logits = self.s * logits

        loss = self.loss_fn(logits, targets)
        return loss

    def get_pred_scores(self, pred):
        pred = F.normalize(pred, p=2, dim=1)
        w = F.normalize(self.w, p=2, dim=0)
        cos_theta = torch.matmul(pred, w)  # 两个向量夹角θ的余弦值
        return cos_theta


def build_accuracy_fn(loss_fn):
    def accuracy(pred, y):
        logits = loss_fn.get_pred_scores(pred)
        _, preds = torch.max(logits, 1)
        return (preds == y).float().mean()

    return accuracy


def training_with_arc_theta():
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

    loss_fn = ArcFaceCrossEntropyLossWithTheta(512, 6)  # 交叉熵损失函数
    metrics = {
        'fps': training.BatchTimer(),  # 效率指标
        'acc': build_accuracy_fn(loss_fn)  # 评估指标
    }

    writer = SummaryWriter(log_dir="./output/v3/summary")
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

    torch.save(resnet.state_dict(), './output/v3/model_param.pkl')


if __name__ == '__main__':
    training_with_arc_theta()
    """
    训练FaceNet三元组损失函数的伪代码:
        # 一、从DataSet中获取anchor、pos、neg对应的图像
            ## self.imgs: 数组，存储的是三元组 (图像数据，图像对应的类别id，该图像是该类别/人的第几个图像)
            ## self.label_to_imgs：字典，图像类别id为key，value为该类别的对应图像(同一个人的多个图像)list
            anchor_img, anchor_person_id, anchor_img_num = self.imgs[anchor_idx]
            pos_img = None
            for _img, _img_num in shuffle(self.label_to_imgs[anchor_person_id]):
                if _img_num != anchor_img_num:
                    pos_img = _img
                    break
            neg_img = None
            neg_person_id = None
            while True:
                neg_label_id = randint(0, len(self.label_to_imgs))
                if neg_label_id != anchor_person_id:
                    break
            for _img, _img_num in shuffle(self.label_to_imgs[neg_person_id]):
                neg_img = _img
                break
            
            return  anchor_img, pos_img, neg_img
        
        # 二、训练的伪代码
            batch_anchor_img = ... [N,3,H,W]
            batch_pos_img = ... [N,3,H,W]
            batch_neg_img = .... [N,3,H,W]
            
            anchor_emb = net(batch_anchor_img) # [N,512] anchor图像对应的L2norm之后的特征向量
            pos_emb = net(batch_pos_img) # [N,512] pos图像对应的L2norm之后的特征向量
            neg_emb = net(batch_neg_img) # [N,512] neg图像对应的L2norm之后的特征向量
            
            anchor_pos_dist = 1 - sum(anchor_emb * pos_emb, 1) # [N]
            anchor_neg_dist = 1 - sum(anchor_emb * neg_emb, 1) # [N]
            loss = anchor_pos_dist - anchor_neg_dist + alpha # [N]
            loss = mean(loss) 
    """
