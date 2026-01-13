# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/6 20:51
Create User : 19410
Desc : xxx

加载VOC的XML格式，解析输出YOLO格式

定义一个voc到yolo数据格式的转换代码
voc数据标定的时候给定的是：左上角、右下角的坐标(xmin ymin xmax ymax)
yolo数据标定的时候给定的是: 中心点坐标、宽度、高度(cx cy w h)，并且是和原始图像width和height的百分比

"""

import os
from tqdm import tqdm
from xml.etree import ElementTree as ET

if __name__ == '__main__':
    # 将数字当成一个类别，字母当成另外一个类别
    labelname2id_mapping = {
        '0': 0, '1': 0, '2': 0, '3': 0, '4': 0,
        '5': 0, '6': 0, '7': 0, '8': 0, '9': 0
    }
    # 每个类别的数字当成一个类别
    # labelname2id_mapping = {
    #     '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    #     '5': 5, '6': 6, '7': 7, '8': 8, '9': 9
    # }
    default_label_id = None
    # voc_label_dir = r"D:\datas\MNIST\1106\voc_labels"
    voc_label_dir = r"D:\datas\MNIST\voc_labels"
    yolo_label_dir = r"D:\datas\MNIST\1106\yolo_labels"
    os.makedirs(yolo_label_dir, exist_ok=True)
    whs = []

    # 读取所有的voc xml文件名称列表
    voc_label_names = os.listdir(voc_label_dir)
    # 遍历处理每个xml文件
    for voc_label_name in tqdm(voc_label_names):
        # 1. 解析xml文件
        tree = ET.parse(os.path.join(voc_label_dir, voc_label_name))
        # 2. 获取得到xml的根节点信息
        root = tree.getroot()
        # 3. 获取图像大小
        size_obj = root.find('size')
        width = float(size_obj.find('width').text)
        height = float(size_obj.find('height').text)

        # 4. 遍历object对象
        yolo_label_file = os.path.join(yolo_label_dir, f"{os.path.splitext(voc_label_name)[0]}.txt")
        with open(yolo_label_file, "w", encoding="utf-8") as writer:
            for obj in root.findall('object'):
                # a. 提取标签名称
                label_name = obj.find('name').text
                # b. 提取左上角、右下角坐标
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)

                # c. 转换成中心点坐标、宽度、高度
                w = xmax - xmin
                h = ymax - ymin
                x = xmin + w / 2.0
                y = ymin + h / 2.0

                # 转换为百分比
                w = w / width
                h = h / height
                x = x / width
                y = y / height

                # 转换输出
                label_id = labelname2id_mapping.get(label_name, default_label_id)
                if label_id is not None:
                    whs.append((xmax - xmin, ymax - ymin))
                    writer.writelines(f"{label_id} {x:.5f} {y:.5f} {w:.5f} {h:.5f}\n")

    # 对边框的高度宽度做一个聚类
    import numpy as np
    from sklearn.cluster import KMeans
    import matplotlib.pyplot   as plt

    whs = np.asarray(whs)
    model = KMeans(n_clusters=9)
    model.fit(whs)
    print(model.cluster_centers_)
    print(whs.min(axis=0))
    print(whs.max(axis=0))

    plt.scatter(whs[:, 0], whs[:, 1])
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='r')
    plt.show()

