# -*- coding: utf-8 -*-
import logging
import os

from PIL.Image import Image

from .face_models import FaceModel
from .face_vectors import VectorService


# noinspection DuplicatedCode
class FaceRunner(object):
    def __init__(self, vector_root_dir, project_root_dir):
        super(FaceRunner, self).__init__()
        self.model = FaceModel(project_root_dir)
        self.vector = VectorService(vector_root_dir=vector_root_dir, project_root_dir=project_root_dir)

    def face_compare(self, user_id: str, image: Image, threshold: float = 0.8, how='any'):
        """
        人证对比逻辑
        """
        try:
            # 1. 参数的检查
            if user_id is None:
                return {'code': 1, 'msg': 'user_id为空'}
            if image is None:
                return {'code': 1, 'msg': '图像对象为空'}
            if how not in ['any', 'all']:
                return {'code': 1, 'msg': '参数how仅允许给定any或者all'}
            is_all = how == 'all'

            # 2. 获取user_id对应的图像特征向量，返回的是一个list
            user_features = self.vector.get_user_features(user_id=user_id)
            if len(user_features) == 0:
                return {'code': 1, 'msg': f'当前用户不存在特征向量 {user_id}'}

            # 3. 从image图像中提取人脸区域对象，并进行检查操作
            num, faces = self.model.get_face_images(image=image, save_path=None)
            if faces is None:
                return {'code': 1, 'msg': f'人脸检测异常，总共检测出:{num}个人脸，请重新给定有效的image图像参数!'}

            # 4. 提取人脸区域对应的特征向量，并进行检查操作
            features = self.model.get_face_features(faces)
            if features is None:
                return {'code': 1, 'msg': f'人脸特征向量提取异常，请重新给定有效的image图像参数!'}

            # 5. 遍历比较特征向量的相似度，进行逻辑判断
            max_sim = float('-inf')
            min_sim = float('inf')
            data_list = []
            for (user_feature, user_face_img_path) in user_features:
                # a. 计算user_feature和feature之间的特征向量的相似度
                sim = self.vector.calc_sim(v1=user_feature, v2=features)
                # b. 过滤
                if sim >= threshold:
                    _data = {'sim': f'{sim:.3f}', 'image_path': user_face_img_path, 'user_id': user_id}
                    if is_all:
                        data_list.append(_data)
                    else:
                        return {
                            'code': 0,
                            'msg': f'人证对比成功!',
                            'data': _data
                        }
                else:
                    min_sim = min(sim, min_sim)
                    max_sim = max(sim, max_sim)

            if is_all:
                if len(data_list) == len(user_features):
                    return {
                        'code': 0,
                        'msg': f'人证对比成功!',
                        'data': data_list
                    }
                else:
                    return {
                        'code': 0,
                        'msg': f'人证对比失败，不是所有图像都满足相似度阈值，最小相似度为:{min_sim:.3f}'
                    }
            else:
                return {
                    'code': 0,
                    'msg': f'人证对比失败，相似度最大为:{max_sim:.3f}'
                }
        except Exception as e:
            logging.error("人证对比逻辑异常", exc_info=e)
            return {
                'code': 1,
                'msg': f'人证对比逻辑异常:{e}'
            }

    def face_search(self, k: int, image: Image, threshold: float = 0.8):
        """
        人脸检索
        """
        try:
            if k <= 0:
                return {'code': 1, 'msg': 'k必须大于0'}
            if image is None:
                return {'code': 1, 'msg': '图像对象为空'}
            # 1. 从image中检测出人脸
            num, faces = self.model.get_face_images(image=image, save_path=None)
            if faces is None:
                return {'code': 1, 'msg': f'人脸检测异常，总共检测出:{num}个人脸，请重新给定有效的image图像参数!'}

            # 2. 提取人脸特征向量
            features = self.model.get_face_features(faces)
            if features is None:
                return {'code': 1, 'msg': f'人脸特征向量提取异常，请重新给定有效的image图像参数!'}

            # 3. 检索
            result = self.vector.search(vector=features, k=k, thred=threshold)
            if len(result) == 0:
                return {'code': 0, 'msg': '检测出人脸，但是没有满足要求的匹配人脸.', 'data': []}

            return {'code': 0, 'msg': f'检测成功，总共匹配人脸数目:{len(result)}', 'data': result}
        except Exception as e:
            logging.error("检测成功，数据库异常.", exc_info=e)
            return {'code': 1, 'msg': f'检测成功，数据库异常:{e}'}

    def face_register(self, user_id: str, image: Image, save_path=None, threshold=0.7):
        """
        人脸注册
        实际的项目中，主要区别包括:
        1. 保存图像可能是保存到一个独立的图像服务器中
        2. 向量检索应该存在一个独立的服务
        """
        try:
            if user_id is None:
                return {'code': 1, 'msg': 'user_id为空'}
            if image is None:
                return {'code': 1, 'msg': '图像对象为空'}

            # 1. 从image中检测出人脸
            num, faces = self.model.get_face_images(image=image, save_path=save_path)
            if faces is None:
                return {'code': 1, 'msg': f'人脸检测异常，总共检测出:{num}个人脸，请重新给定有效的image图像参数!'}

            # 2. 提取人脸特征向量
            features = self.model.get_face_features(faces)
            if features is None:
                os.remove(save_path)
                return {'code': 1, 'msg': f'人脸特征向量提取异常，请重新给定有效的image图像参数!'}

            # 3. 需要检查user_id已有的特征向量和准备新增的特征向量之间的最小相似度是否满足阈值
            user_features = self.vector.get_user_features(user_id=user_id)
            if len(user_features) > 0:
                min_sim = float('inf')
                for (user_feature, user_face_img_path) in user_features:
                    # a. 计算user_feature和feature之间的特征向量的相似度
                    sim = self.vector.calc_sim(v1=user_feature, v2=features)
                    # b. 获取最小相似度
                    min_sim = min(sim, min_sim)
                if min_sim < threshold:
                    os.remove(save_path)
                    return {
                        'code': 1,
                        'msg': f'无法录入该用户的新人脸图像，请重新拍照，当前录入图像和之前图像之间的相似度低于阈值:{min_sim:.3f}'
                    }

            # 4. 信息保存
            self.vector.add_embedding(user_id=user_id, vector=features, save_path=save_path)

            return {'code': 0, 'msg': '录入成功'}
        except Exception as e:
            if os.path.exists(save_path):
                os.remove(save_path)
            logging.error("录入失败，数据库异常.", exc_info=e)
            return {'code': 1, 'msg': f'录入失败，数据库异常:{e}'}
