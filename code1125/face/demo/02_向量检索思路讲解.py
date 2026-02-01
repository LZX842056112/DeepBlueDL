# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/11/23 10:28
Create User : 19410
Desc : 向量检索 --> 从海量向量数据库中找出到当前向量相似的k个向量

向量索引库功能：提供基于向量相似度进行提取最相似的向量列表
pip install faiss-cpu==1.11.0
"""

import time

import faiss
import numpy as np

if __name__ == '__main__':
    k = 5
    n = 100000
    dimension = 128
    emb_tables = np.random.randn(n, dimension)  # 向量数据库
    emb_tables = emb_tables / np.sqrt(np.sum(np.power(emb_tables, 2), axis=1, keepdims=True))
    emb_tables = emb_tables.astype(np.float32)  # 类型转换为float32 ---> faiss框架只支持float32
    print(emb_tables.shape)

    rnd_idx = np.random.randint(low=0, high=n)
    query_emb = emb_tables[rnd_idx:rnd_idx + 1]
    # query_emb = np.random.randn(1, dimension)
    # query_emb = query_emb.astype(np.float32) # 类型转换为float32 ---> faiss框架只支持float32
    query_emb = query_emb / np.sqrt(np.sum(np.power(query_emb, 2), axis=1, keepdims=True))
    print(rnd_idx, query_emb.shape)


    def sim_func(_v1, _v2):
        # return np.sum(_v1 * _v2) / (np.sqrt(np.power(_v1, 2).sum()) * np.sqrt(np.power(_v2, 2).sum()))
        return np.sum(_v1 * _v2)


    # 暴力方式：获取最相似的k个向量 --> 首先需要获取每个向量和当前向量的相似度，然后排序获取最大的前k个
    total_sims = []
    st = time.time()
    print(f"开始暴力求解:{st}")
    for i, emb in enumerate(emb_tables):
        total_sims.append((i, sim_func(emb, query_emb)))
    # print(total_sims)
    total_sims = sorted(total_sims, key=lambda t: t[1], reverse=True)
    final_sims = total_sims[:k]
    print(f"暴力求解结束: {time.time() - st}")
    print(list(map(lambda t: (t[0], float(f"{t[1]:.2f}")), final_sims)))

    # 基于Faiss框架的向量检索
    ## 第一步: 构建向量索引库 --> 在内存提前构建好一个对象
    index = faiss.index_factory(
        dimension,  # 维度大小
        "HNSW8",  # 向量库的类型 --> 算法名称
        # "HNSW512",  # 向量库的类型 --> 算法名称
        faiss.METRIC_INNER_PRODUCT  # 给定距离的表达形式/相似度的表达形式
    )
    index.train(emb_tables)
    index.add(emb_tables)

    ## 第二步：检索
    st = time.time()
    print(f"Faiss检索开始:{st}")
    D, I = index.search(query_emb, k)  # D是距离矩阵，I是索引矩阵
    print(f"Faiss检索结束: {time.time() - st}")
    print(f"相似的距离信息:{D}")
    print(f"相似的样本index:{I}")
