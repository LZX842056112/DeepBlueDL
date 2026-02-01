# -*- coding: utf-8 -*-
import os

import torch

from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from facenet_pytorch.models.mtcnn import MTCNN


class FaceModel(object):
    def __init__(self, project_root_dir):
        super(FaceModel, self).__init__()
        device = torch.device("cpu")
        self.mtcnn = MTCNN(
            image_size=160,  # 输出人脸图像的区域大小，eg: 最终输出160*160
            margin=16,  # 在截取人脸区域的时候，往外扩展多少个像素
            min_face_size=20,  # 最小的人脸大小
            thresholds=[0.6, 0.7, 0.8],  # P-、R-、O-Net三个网络后处理时候的阈值
            factor=0.709,  # 图像金字塔的缩放比例
            post_process=True,
            select_largest=True,
            selection_method=None,
            keep_all=True,  # 是否保留所有检测图像
            device=device
        )
        # 为了减少模型下载，这里指定模型路径
        os.environ['TORCH_HOME'] = os.path.join(project_root_dir, "static", 'models')
        self.feature_net = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    @torch.no_grad()
    def get_face_images(self, image, save_path=None):
        # Detect faces 检测人脸
        batch_boxes, _, _ = self.mtcnn.detect(image, landmarks=True)
        face_number_with_detect = len(batch_boxes)
        if face_number_with_detect == 0:
            return face_number_with_detect, None  # 未检测出人脸区域
        if face_number_with_detect > 1:
            return face_number_with_detect, None  # 检测出多个人脸区域

        # 提取人脸区域并保存
        faces = self.mtcnn.extract(image, batch_boxes, save_path)
        return face_number_with_detect, faces

    @torch.no_grad()
    def get_face_features(self, image):
        dim = image.dim()
        if dim == 3:
            image = torch.unsqueeze(image, 0)
            return self.feature_net(image).numpy()[0]
        elif dim == 4:
            features = self.feature_net(image).numpy()
            if len(image) == 1:
                return features[0]
            else:
                return features
        else:
            return None
