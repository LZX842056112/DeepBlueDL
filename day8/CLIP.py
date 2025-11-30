import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from transformers import BertModel, BertTokenizer
import timm
import numpy as np


# 图像编码器 - 使用ViT
class ViT(nn.Module):
    def __init__(self, output_dim):
        super(ViT, self).__init__()
        # 使用来自timm的ViT模型
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=output_dim)

    def forward(self, x):
        return self.vit(x)


# 文本编码器 - 使用BERT
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()

        BERT_LOCAL_PATH = './bert-base-uncased'
        self.model = BertModel.from_pretrained(BERT_LOCAL_PATH)
        self.tokenizer = BertTokenizer.from_pretrained(BERT_LOCAL_PATH)

    def forward(self, texts):
        # 文本通过BERT
        encoded_input = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**encoded_input)
        return outputs.last_hidden_state[:, 0, :]


# 加载CIFAR10数据集
def load_cifar10_dataset():
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_dataset = CIFAR10(root='./cifar10', train=True, download=True, transform=transform)
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    classes = train_dataset.classes
    return loader, classes


class CLIP(nn.Module):
    def __init__(self, image_output_dim, text_output_dim):
        super(CLIP, self).__init__()
        self.image_encoder = ViT(image_output_dim)
        self.text_encoder = TextEncoder()

        # 因为图像和文本emb可能维度不同(图像512，文本768)，所以需要对图像和文本的emb再经过一层以将维度持平
        self.W_i = nn.Parameter(torch.randn(image_output_dim, text_output_dim))
        self.W_t = nn.Parameter(torch.randn(768, text_output_dim))  # BERT-base的最后隐藏层大小为768

    def forward(self, images, texts):
        I_f = self.image_encoder(images)  # (B,3,224,224) -> (B, 512)
        T_f = self.text_encoder(texts)  # （B）-> (B, 768)

        # 调整维度
        I_e = torch.matmul(I_f, self.W_i)  # (B, 512)
        T_e = torch.matmul(T_f, self.W_t)  # (B, 512)

        # 计算余弦相似度
        logits = torch.matmul(I_e, T_e.T)  # (B,B)
        return logits


# 主函数
def main():
    # 加载数据集
    dataset, classes = load_cifar10_dataset()
    clip_model = CLIP(image_output_dim=512, text_output_dim=512)

    for images, labels in dataset:
        # 获取一个小批量的图像和标签
        texts = [classes[label] for label in labels]

        logits = clip_model(images, texts)  # (B,B)
        labels = torch.arange(logits.shape[0])  # (0,1,2,3)

        # 计算损失 loss_i是每一张图像我都要把它判定为正确得文本，而loss_t是每一个文本我都要把它判定为正确得图像
        loss_i = torch.nn.CrossEntropyLoss()(logits, labels)
        loss_t = torch.nn.CrossEntropyLoss()(logits.T, labels)
        loss = (loss_i + loss_t) / 2

        # 输出损失
        print(loss)


if __name__ == "__main__":
    main()
