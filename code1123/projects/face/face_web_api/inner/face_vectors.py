# -*- coding: utf-8 -*-
"""
TODO: 检索最终的相关内容可以看推荐部分的内容，这里直接copy
"""

import os

import pickle

# conda install -c pytorch faiss-cpu 或者 pip install  faiss-cpu
# 1.8.0 faiss就是一个向量检索服务，可以非常方便的获取最相似的特征向量
import faiss
import numpy as np


class VectorService(object):
    def __init__(self, vector_root_dir, project_root_dir):
        super(VectorService, self).__init__()
        self.vector_root_dir = vector_root_dir
        self.project_root_dir = project_root_dir
        os.makedirs(self.vector_root_dir, exist_ok=True)
        self.dims = 512

        self.embedding_file = os.path.join(self.vector_root_dir, "emb.pkl")
        if os.path.exists(self.embedding_file):
            self.embeddings = np.asarray(self.load_pickle_data(self.embedding_file)).astype('float32')
        else:
            self.embeddings = np.zeros((0, self.dims), dtype='float32')

        # 一种Faiss向量索引方式
        self.face_index = faiss.index_factory(self.dims, 'HNSW16', faiss.METRIC_INNER_PRODUCT)
        self.face_index.add(self.embeddings)

        self.face_index_id_2_user_id = {}  # face_index中的第一个向量对应的实际用户id是哪一个
        self.idx2_user_id_file = os.path.join(self.vector_root_dir, "user_id.pkl")
        if os.path.exists(self.idx2_user_id_file):
            self.face_index_id_2_user_id = self.load_pickle_data(self.idx2_user_id_file)
        self.user_id_2_face_index_id = {}  # 保存的是user_id对应向量保存到第几行以及路径
        for face_index_id in self.face_index_id_2_user_id.keys():
            user_id, save_path = self.face_index_id_2_user_id[face_index_id]
            if user_id not in self.user_id_2_face_index_id:
                self.user_id_2_face_index_id[user_id] = []
            self.user_id_2_face_index_id[user_id].append((face_index_id, save_path))

    def add_embedding(self, user_id, vector, save_path):
        vector = np.asarray(vector, dtype='float32').reshape((1, self.dims))
        self.embeddings = np.concatenate([self.embeddings, vector], axis=0)  # 当前人脸向量追加到最后
        self.face_index.add(vector)  # 索引追加
        self.save_pickle_data(self.embedding_file, self.embeddings)  # 保存更新
        face_index_id = len(self.face_index_id_2_user_id)
        save_path = save_path.replace(self.project_root_dir, ".").replace("\\", "/")  # NOTE: 保存的时候可以将盘符去掉
        self.face_index_id_2_user_id[face_index_id] = [user_id, save_path]
        if user_id not in self.user_id_2_face_index_id:
            self.user_id_2_face_index_id[user_id] = []
        self.user_id_2_face_index_id[user_id].append((face_index_id, save_path))
        self.save_pickle_data(self.idx2_user_id_file, self.face_index_id_2_user_id)

    def search(self, vector, k, thred=0.8):
        vector = np.asarray(vector, dtype='float32').reshape((1, self.dims))
        prob, idx = self.face_index.search(vector, k)  # 获取最匹配的向量对应索引以及相似度
        prob = prob[0]  # [k]
        idx = idx[0]  # [k]
        # 最终结果的拼接
        result = []
        for i in range(len(idx)):
            j = idx[i]
            if j == -1:
                break
            if prob[i] < thred:
                break
            user_id, image_path = self.face_index_id_2_user_id[j]
            result.append({
                'user_id': user_id,
                'image_path': image_path,
                'prob': f'{prob[i]:.3f}'
            })
        return result

    def get_user_features(self, user_id):
        # 获取user_id对应的所有人脸向量行号以及对应的图像img_path
        _list = self.user_id_2_face_index_id.get(user_id)
        if _list is None:
            return []
        # 将行号转换为特征向量
        _list = [(self.embeddings[i], img_path) for (i, img_path) in _list]
        return _list

    @staticmethod
    def calc_sim(v1, v2):
        v1 = np.reshape(v1, -1)
        v2 = np.reshape(v2, -1)
        if len(v1) != len(v2):
            raise ValueError(f"无法计算两个维度大小不一致的向量的相似度 {len(v1)} -- {len(v2)}")
        return np.sum(v1 * v2)

    @staticmethod
    def load_pickle_data(path):
        with open(path, 'rb') as r:
            return pickle.load(r)

    @staticmethod
    def save_pickle_data(path, data):
        with open(path, 'wb') as w:
            pickle.dump(data, w)
