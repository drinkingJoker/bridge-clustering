'''
Author       : Leaf(2064944038@qq.com)
Version      : V1.0
Date         : 2025-03-01 10:00:00
Description  : 使用KMeans算法进行图像分割
'''
import os
import sys

import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

cur_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(cur_dir)
sys.path.append(parent_dir)

def generate_distinct_colors(n_clusters):
    """
    生成鲜艳的颜色列表，用于聚类结果可视化
    
    参数:
        n_clusters: 聚类数量
        
    返回:
        colors: 颜色数组，形状为 (n_clusters, 3)，RGB格式
    """
    # 使用HSV颜色空间生成均匀分布的颜色
    colors = []
    for i in range(n_clusters):
        # 在HSV空间中，H值决定颜色，均匀分布可以得到不同的颜色
        h = i / n_clusters
        # 饱和度和亮度设置为较高的值，使颜色鲜艳
        s = 0.8
        v = 0.9
        # 转换为RGB
        r, g, b = plt.cm.hsv(h)[0:3]
        colors.append([int(r*255), int(g*255), int(b*255)])
    return np.array(colors, dtype=np.uint8)

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
    # 生成鲜艳的颜色
    distinct_colors = generate_distinct_colors(n_clusters)
    
    # 创建分割结果（使用鲜艳的颜色而不是原图像的颜色）
    height, width = img.shape[:2]
    segmented_img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(n_clusters):
        segmented_img[labels.reshape(height, width) == i] = distinct_colors[i]
    
    return segmented_img


if __name__ == '__main__':
    datas_dir = parent_dir + '/data/Berkeley Segmentation Dataset 500'
    images_path = datas_dir + '/images'
    trains_path = images_path + '/train'
    select_dir = trains_path

    save_path = './result/'
    clusters_savepath = save_path + 'kmeans_segmentation/'
    # 确保保存路径存在
    os.makedirs(clusters_savepath, exist_ok=True)
    plot_config = {'bbox_inches': 'tight', 'pad_inches': 0}
    
    # 定义要生成的聚类数量列表
    cluster_numbers = [2, 4, 8]
    
    for image in os.listdir(select_dir):
        image_path = select_dir + '/' + image

        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            continue

        # 图像转换为一维数据
        data = img.reshape((-1, 3)).astype(np.float32)

        # 为每个聚类数量单独执行KMeans聚类
        segmented_images = []
        for n_clusters in cluster_numbers:
            # 创建KMeans聚类实例并执行聚类
            kmeans = KMeans(n_clusters=n_clusters)
            labels = kmeans.fit_predict(data)
            
            # 生成分割图像
            segmented_img = generate_segmented_image(data, img, labels, n_clusters)
            segmented_images.append(segmented_img)

        # 转换原始图像为RGB显示
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 准备显示图像列表
        images = [img] + segmented_images

        # 用来正常显示中文标签
        plt.rcParams['font.sans-serif'] = ['SimHei']
        
        # 准备标题列表
        titles = [u'原始图像'] + [f'n_clusters={n}' for n in cluster_numbers]
        
        # 显示图像
        fig = plt.figure(figsize=(10, 10))
        for i in range(len(images)):
            plt.subplot(2, 2, i + 1)
            plt.imshow(images[i])
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        
        plt.tight_layout()
        # plt.show()
        plt.close(fig)
        
        # 保存结果图像
        cluster_savepath = f'{clusters_savepath}{image}.png'
        fig.savefig(cluster_savepath, **plot_config)
        print(f"已保存分割结果: {cluster_savepath}")