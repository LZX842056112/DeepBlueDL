# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/23 15:29
Create User : 19410
Desc : xxx
"""
import os

import torch
import numpy as np
from torchvision import transforms
from PIL import Image

from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from facenet_pytorch.models.mtcnn import fixed_image_standardization


@torch.no_grad()
def t0():
    device = torch.device("cpu")
    feature_net = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    feature_net.eval()
    print(feature_net)
    to_tensor = transforms.Compose([
        transforms.ToTensor()
    ])

    img = Image.open(r"./output/mtcnn/faces2_png/faces2_4.png")
    img = to_tensor(img)
    img = img[None]  # [C,H,W] -> [1,C,H,W]

    features = feature_net(img)
    print(features.shape)
    print(features)


@torch.no_grad()
def t1():
    device = torch.device("cpu")
    feature_net = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    feature_net.eval()

    # 模型恢复 --> 模型的模型参数是基于分类loss进行训练的
    # param = torch.load("./output/v2/model_param.pkl")
    # 模型恢复 --> 模型参数是基于ArcFace 进行训练的
    param = torch.load("./output/v3/model_param.pkl")

    for _del_key in ['logits.bias', 'logits.weight']:
        if _del_key in param:
            del param[_del_key]
    missing_keys, unexpected_keys = feature_net.load_state_dict(param, strict=False)
    print(f"missing_keys {len(missing_keys)}")
    print(f"unexpected_keys {len(unexpected_keys)} {unexpected_keys}")
    print("=" * 50)

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        # lambda t: torch.permute(torch.tensor(np.asarray(t), dtype=torch.float32), dims=(2, 0, 1)),
        # fixed_image_standardization
    ])

    def _fetch_features(_file):
        img = Image.open(_file)
        img = to_tensor(img)
        img = img[None]  # [C,H,W] -> [1,C,H,W]

        return feature_net(img)[0].numpy()

    img_data_dir = "./datas/LFW/images_cropped"
    img_features = []
    img_labels = []
    person_names = os.listdir(img_data_dir)
    for pname in person_names:
        p_file = os.path.join(img_data_dir, pname)
        img_names = os.listdir(p_file)
        for img_name in img_names:
            img_file = os.path.join(p_file, img_name)
            img_feature = _fetch_features(img_file)
            img_features.append(img_feature)
            if pname in img_name:
                img_labels.append(img_name)
            else:
                img_labels.append(f"{pname}_{img_name}")
            # img_labels.append((pname, img_name))

    img_features = np.stack(img_features, axis=0)
    print(img_features.shape)

    # 计算相似度
    bs, e = img_features.shape

    def sim_func(_v1, _v2):
        return np.sum(_v1 * _v2) / (np.sqrt(np.power(_v1, 2).sum()) * np.sqrt(np.power(_v2, 2).sum()))

    for i in range(bs):
        print("=" * 50)
        vi = img_features[i]
        for j in range(bs):
            vj = img_features[j]
            sim_ij = sim_func(vi, vj)
            print(f"图像 {i}_{img_labels[i]} 和图像 {j}_{img_labels[j]} 之间的相似度为: {sim_ij:.3f}")


if __name__ == '__main__':
    t1()
