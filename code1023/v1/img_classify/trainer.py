# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/10/14 20:17
Create User : 19410
Desc : 训练器
"""
import os
from datetime import datetime
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# noinspection PyTypeChecker
class Trainer:
    def __init__(self,
                 net: nn.Module, loss_fn: nn.Module, train_opt: optim.Optimizer,
                 train_dataloader: DataLoader, val_dataloader: DataLoader,
                 total_epoch: int, dump_model_dir: str, class_names: List[str],
                 device=None, summary_writer: SummaryWriter = None):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"当前运行采用默认设备:{device}")

        self.device = device
        self.net = net
        self.loss_fn = loss_fn
        self.train_opt: optim.Optimizer = train_opt
        self.total_epoch = total_epoch
        self.dump_model_dir = dump_model_dir
        self.class_names = class_names

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.best_eval_acc = 0.0

        # 可视化对象保存
        self.summary_steps = {}
        self.summary_writer = summary_writer

        # 参数恢复
        self.reload()

        # device设备转换
        self.net.to(device=self.device)
        self.loss_fn.to(device=self.device)

    def summary_scalar_logs(self, logs):
        if self.summary_writer is None:
            return

        # 针对待存储的数据进行遍历存储
        for log_key, log_value in logs:
            _step = self.summary_steps.get(log_key) or 1
            self.summary_writer.add_scalar(log_key, log_value, global_step=_step)
            self.summary_steps[log_key] = _step + 1

    def train_epoch(self, epoch):
        self.net.train()
        for batch_idx, batch in enumerate(self.train_dataloader):
            batch_images = batch['image'].to(device=self.device)
            batch_labels = batch['label'].to(device=self.device)

            # b. 前向过程执行
            scores = self.net(batch_images)  # 模型的前向执行，得到每个样本属于各个类别的预测置信度 [N,num_classes]
            loss = self.loss_fn(scores, batch_labels)

            # c. 反向过程执行
            loss.backward()  # 基于loss求解每个参数对应的梯度值
            self.train_opt.step()  # 当前train_opt中对应的所有可训练参数进行对应的更新(基于梯度值)
            self.train_opt.zero_grad()  # 将当前train_opt中包含的所有参数的对应梯度值重置为0

            print(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item():.3f}")

            self.summary_scalar_logs(logs=[('train_batch_loss', loss.item())])

    def eval_epoch(self, epoch):
        self.net.eval()  # 标记当前模型进入评估执行逻辑
        with torch.no_grad():
            all_predicts, all_trues = [], []
            for val_batch_idx, batch in enumerate(self.val_dataloader):
                val_batch_images = batch['image'].to(device=self.device)
                val_batch_labels = batch['label'].to(device=self.device)

                # b. 前向过程执行
                val_scores = self.net(val_batch_images)  # 模型的前向执行，得到每个样本属于各个类别的预测置信度 [N,num_classes]
                val_loss = self.loss_fn(val_scores, val_batch_labels)

                # 当前批次的预测标签id和实际标签id保存
                all_predicts.append(torch.argmax(val_scores.cpu(), dim=1).numpy())
                all_trues.append(val_batch_labels.cpu().numpy())

                print(f"Eval Epoch {epoch} Batch {val_batch_idx} Loss {val_loss.item():.3f}")
                self.summary_scalar_logs(logs=[('eval_batch_loss', val_loss.item())])

            # 合并所有的
            all_predicts = np.concatenate(all_predicts, axis=0)
            all_trues = np.concatenate(all_trues, axis=0)

            val_acc = metrics.accuracy_score(all_trues, all_predicts)
            val_report = metrics.classification_report(all_trues, all_predicts)
            print("=" * 50)
            print(f"Eval Epoch {epoch} Accuracy {val_acc:.3f}")
            print(f"Eval Epoch {epoch} Class report:\n{val_report}\n")
            print("=" * 50)
            self.summary_scalar_logs(logs=[
                ('eval_epoch_acc', val_acc),
                ('eval_recall', metrics.recall_score(all_trues, all_predicts, average='micro')),
                ('eval_precision', metrics.precision_score(all_trues, all_predicts, average='micro'))
            ])

        return val_acc

    def reload(self):
        """
        参数恢复
        :return:
        """
        ckpt_file = os.path.join(self.dump_model_dir, 'best.pkl')
        if os.path.exists(ckpt_file):
            print(f"参数恢复:{ckpt_file}")
            ckpt = torch.load(ckpt_file, map_location='cpu')
            missing_keys, unexpected_keys = self.net.load_state_dict(ckpt['net_param'], strict=False)
            print(f"模型参数恢复:\n\tmissing_keys: {missing_keys}\n\tunexpected_keys: {unexpected_keys}\n")
            missing_keys, unexpected_keys = self.loss_fn.load_state_dict(ckpt['loss_param'], strict=False)
            print(f"损失函数参数恢复:\n\tmissing_keys: {missing_keys}\n\tunexpected_keys: {unexpected_keys}\n")
            self.train_opt.load_state_dict(ckpt['opt_param'])
            if 'acc' in ckpt:
                self.best_eval_acc = ckpt['acc']
            if 'summary_steps' in ckpt:
                self.summary_steps = ckpt['summary_steps']

    def save(self, epoch, epoch_eval_acc):
        obj = {
            'net_param': self.net.state_dict(),  # 将模型的参数以dict的形式保存下来
            'loss_param': self.loss_fn.state_dict(),
            'opt_param': self.train_opt.state_dict(),
            'date': datetime.now(),
            "class_names": self.class_names,
            'epoch': epoch,
            'acc': epoch_eval_acc,
            'summary_steps': self.summary_steps
        }

        os.makedirs(self.dump_model_dir, exist_ok=True)
        # torch.save底层是基于python的 pickle 库进行持久化操作；对应的加载API为 torch.load
        torch.save(obj, os.path.join(self.dump_model_dir, 'last.pkl'))
        if epoch_eval_acc > self.best_eval_acc:
            print(f"Epoch {epoch} 最优模型持久化 {self.best_eval_acc:.4f} - {epoch_eval_acc:.4f}")
            self.best_eval_acc = epoch_eval_acc
            torch.save(obj, os.path.join(self.dump_model_dir, 'best.pkl'))

    def train(self):
        for epoch in range(self.total_epoch):
            # 进入当前epoch的模型训练
            self.train_epoch(epoch)

            # 模型评估
            epoch_eval_acc = self.eval_epoch(epoch)

            # 模型持久化保存
            self.save(epoch, epoch_eval_acc)

        # 日志输出对象的close操作
        if self.summary_writer is not None:
            self.summary_writer.close()
            if hasattr(self.summary_writer, 'hook_handles'):
                for hook in getattr(self.summary_writer, 'hook_handles'):
                    hook.remove()
