# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/27 21:33
Create User : 19410
Desc : 关键点数据转换
"""
# -*- coding: utf-8 -*-
"""
将标注好的json格式数据转换为关键点检测的格式
"""
import json
import os
import numpy as np

root_dir = r"D:\datas\plate_images\keypoints\keypoints_json"
output_dir = r"D:\datas\plate_images\keypoints\labels2"
os.makedirs(output_dir, exist_ok=True)

names = os.listdir(root_dir)
label_name2idx = {}
for name in names:
    json_file = os.path.join(root_dir, name)
    with open(json_file, 'r', encoding='utf-8') as json_reader:
        json_data = json.load(json_reader)
    # print(json_data)

    # 开始提取数据
    h = json_data['imageHeight']
    w = json_data['imageWidth']
    shapes = json_data['shapes']
    if len(shapes) == 0:
        continue
    results = []
    for obj_shape in shapes:
        obj_label = obj_shape['label']  # 取值只有一种: 关键点
        if obj_label in ['lb', 'lt', 'rb', 'rt']:
            # 关键点坐标
            kx1, ky1 = obj_shape['points'][0]
            kx1 = kx1 / w
            ky1 = ky1 / h
            results.append((obj_label, f"{kx1:.5f} {ky1:.5f} 2", kx1, ky1))
        else:
            raise ValueError(f"当前不支持该类别的标签:{obj_shape}")

    # 输出转换
    with open(os.path.join(output_dir, f"{os.path.splitext(name)[0]}.txt"), "w", encoding="utf-8") as writer:
        if len(results) == 4:
            # 按照lb(左下)、lt(左上)、rb(右下)、rt(右上)的顺序排列
            results = sorted(results, key=lambda t: t[0])
            points = ' '.join(list(map(lambda t: t[1], results)))
            box = np.asarray(list(map(lambda t: t[2:], results)))
            x1, y1 = np.min(box[:, 0]), np.min(box[:, 1])
            w = np.max(box[:, 0]) - x1
            h = np.max(box[:, 1]) - y1
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            writer.writelines(f"0 {cx:.5f} {cy:.5f} {w:.5f} {h:.5f} {points}\n")





