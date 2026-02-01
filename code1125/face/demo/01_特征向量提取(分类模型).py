# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/22 16:41
Create User : 19410
Desc : 基于分类模型(分类目标构造的模型)
"""
import os

import numpy as np
import torch


@torch.no_grad()
def t0():
    import classify_17flowers.v1.img_classify.predictor as predictor
    p = predictor.Predictor(
        model_file=os.path.join(predictor.__file__, "..", "..", "deploy", "best.pkl")
    )

    img_data_dir = "./datas/17flowers"
    img_features = []
    img_names = os.listdir(img_data_dir)
    for name in img_names:
        img_file = os.path.join(img_data_dir, name)
        img_feature = p.fetch_features(img_file)
        img_features.append(img_feature)
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
            print(f"图像 {i}_{img_names[i]} 和图像 {j}_{img_names[j]} 之间的相似度为: {sim_ij:.3f}")


@torch.no_grad()
def t1():
    import classify_17flowers.v1.img_classify.predictor as predictor
    p = predictor.Predictor(
        model_file=os.path.join(predictor.__file__, "..", "..", "deploy", "best.pkl")
    )

    img_data_dir = "./datas/LFW/images_cropped"
    img_features = []
    img_labels = []
    person_names = os.listdir(img_data_dir)
    for pname in person_names:
        p_file = os.path.join(img_data_dir, pname)
        img_names = os.listdir(p_file)
        for img_name in img_names:
            img_file = os.path.join(p_file, img_name)
            img_feature = p.fetch_features(img_file)
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
