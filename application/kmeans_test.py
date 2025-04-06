'''
Author       : Leaf(2064944038@qq.com)
Version      : V1.0
Date         : 2025-02-28 13:36:38
Description  : 
'''
import os
import sys

cur_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(cur_dir)
sys.path.append(parent_dir)
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

datas_dir = parent_dir + '/data/Berkeley Segmentation Dataset 500'
images_path = datas_dir + '/images'
trains_path = images_path + '/train'
select_dir = trains_path
from bridge_clustering import BridgeClustering

kls = LocalOutlierFactor
args = {'contamination': .2, 'n_neighbors': 15}
k = 7
for image_name in os.listdir(select_dir):
    print(image_name)
    image_path = select_dir + '/' + image_name

    # 读取图像
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # 将图像转换为特征矩阵
    # 每个像素的特征为 [B, G, R, x, y]
    pixels = image.reshape(-1, 3)  # 提取 BGR 颜色值
    x = np.arange(w).repeat(h).reshape(-1, 1)  # 生成 x 坐标
    y = np.tile(np.arange(h), w).reshape(-1, 1)  # 生成 y 坐标
    features = np.hstack([pixels, x, y])  # 合并 BGR 和坐标

    # 归一化特征
    features = features.astype(np.float32)
    features[:, :3] /= 255.0  # 将 BGR 归一化到 [0, 1]
    features[:, 3:] /= max(h, w)  # 将坐标归一化到 [0, 1]

    # K-Means 聚类
    n_clusters = 4  # 假设分割为 4 个区域
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(features)
    # od = kls(**args)
    # clf = BridgeClustering(od, k, n_clusters)
    # labels_ = clf.fit_predict(features)

    # 生成分割结果
    # 计算每个簇的 BGR 均值
    # cluster_colors_ = np.array([
    #     np.mean(features[labels == i, :3], axis=0) for i in range(n_clusters)
    # ])  # 等价于下面的取 BGR 颜色值
    # print(cluster_colors_)
    cluster_colors_ = np.random.randint(0,
                                        255,
                                        size=(n_clusters, 3),
                                        dtype=np.uint8)
    segmented_ = cluster_colors_[labels]
    segmented_ = (segmented_ * 255).astype(np.uint8)  # 反归一化
    segmented_ = segmented_.reshape(h, w, 3)  # 恢复图像形状

    # cluster_colors = np.array([
    #     np.mean(features[labels == i, :3], axis=0) for i in range(n_clusters)
    # ])  # 等价于下面的取 BGR 颜色值
    # print(cluster_colors)
    segmented = model.cluster_centers_[labels, :3]  # 取 BGR 颜色值
    segmented = (segmented * 255).astype(np.uint8)  # 反归一化
    segmented = segmented.reshape(h, w, 3)  # 恢复图像形状

    # 显示结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title(f"{image_name}")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 2)
    plt.title("k_means")
    plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 3)
    plt.title("clf")
    plt.imshow(cv2.cvtColor(segmented_, cv2.COLOR_BGR2RGB))
    plt.show()
