'''
Author       : Leaf(2064944038@qq.com)
Version      : V1.0
Date         : 2025-01-26 14:55:25
Description  : 
'''
import os
import sys

from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans

cur_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(cur_dir)
sys.path.append(parent_dir)

from utils.function import merge_clusters
from utils.data_utils import read_image
from bridge_clustering import BridgeClustering


def generate_segmented_image(data, img, labels, n_clusters):
    """
    根据聚类标签生成分割图像
    
    参数:
        data: 图像数据（一维数组）
        img: 原始图像
        labels: 聚类标签
        n_clusters: 聚类数量
    
    返回:
        segmented_img: 分割后的图像
    """
    # 合并聚类
    merged_labels = merge_clusters(labels, n_clusters)
    
    # 计算每个聚类的中心点颜色
    centers = np.array([np.mean(data[merged_labels == i], axis=0) for i in range(n_clusters)])
    centers = np.uint8(centers)
    
    # 生成分割图像
    res = centers[merged_labels]
    segmented_img = res.reshape(img.shape)
    
    # 转换为RGB显示
    return cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    datas_dir = parent_dir + '/data/Berkeley Segmentation Dataset 500'
    images_path = datas_dir + '/images'
    trains_path = images_path + '/train'
    select_dir = trains_path

    save_path = './result/'
    clusters_savepath = save_path + 'image_segmentation/'
    plot_config = {'bbox_inches': 'tight', 'pad_inches': 0}

    kls = LocalOutlierFactor
    args = {'contamination': .2, 'n_neighbors': 15}
    k = 7
    
    # 定义要生成的聚类数量列表
    cluster_numbers = [2, 4, 8, 16, 32, 64, 128]
    
    for image in os.listdir(select_dir):
        image_path = select_dir + '/' + image
        # image_path = trains_path + '/' + os.listdir(select_dir)[0]
        # img, data, height, width = read_image(image_path)

        # 读取图像
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 图像转换为一维数据
        data = img.reshape((-1, 3)).astype(np.float32)

        # 创建桥点聚类实例并执行聚类
        od = kls(**args)
        clf = BridgeClustering(od, k)
        labels = clf.fit_predict(data)

        # 生成不同聚类数量的分割图像
        segmented_images = []
        for n_clusters in cluster_numbers:
            segmented_img = generate_segmented_image(data, img, labels, n_clusters)
            segmented_images.append(segmented_img)

        # 转换原始图像为RGB显示
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 准备显示图像列表
        images = [img] + segmented_images

        # 用来正常显示中文标签
        plt.rcParams['font.sans-serif'] = ['SimHei']
        
        # 准备标题列表
        titles = [u'原始图像'] + [f'聚类图像 K={n}' for n in cluster_numbers]
        
        # 显示图像
        fig = plt.figure()
        for i in range(len(images)):
            plt.subplot(2, 4, i + 1),
            plt.imshow(images[i], 'gray'),
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()
        plt.close(fig)
        cluster_savepath = f'{clusters_savepath}{image}.png'
        fig.savefig(cluster_savepath, **plot_config)
