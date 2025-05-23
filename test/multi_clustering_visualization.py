#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author       : Leaf(2064944038@qq.com)
Version      : V1.0
Date         : 2025-01-11 17:27:07
Description  : 多种聚类算法结果可视化比较
'''
import os
import sys
import matplotlib
# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 获取当前文件所在的目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上级目录的绝对路径
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

import pandas as pd
import numpy as np
from matplotlib import gridspec, pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, OPTICS
import hdbscan

from bridge_clustering import BridgeClustering
from utils.border_peel.border_peeling import BorderPeel as BorderPeeling
from utils.autoclust import AUTOCLUST
from utils.function import compute_neighbors, determine_bridges
from utils.data_utils import read_arff, read_dataset


def plot_multi_clustering(filename, X, true_labels):
    """
    在同一张图上展示多种聚类算法的结果
    
    参数:
    filename: 数据集文件名
    X: 数据集特征
    true_labels: 真实的聚类标签
    """
    # 设置不同的聚类算法
    # 1. 桥点聚类 (Bridge Clustering)
    kls = LocalOutlierFactor
    args = {'contamination': .2, 'n_neighbors': 15}
    k = 7
    od = kls(**args)
    bridge_clf = BridgeClustering(od, k)
    bridge_clf.fit(X)
    bridge_labels = bridge_clf.labels_
    
    # 2. HDBSCAN
    hdbscan_clf = hdbscan.HDBSCAN(min_cluster_size=5)
    hdbscan_labels = hdbscan_clf.fit_predict(X)
    
    # 3. OPTICS
    # 计算样本数的7%作为min_samples参数
    min_samples = max(int(X.shape[0] * 0.07), 2)  # 确保至少为2
    optics = OPTICS(min_samples=min_samples)
    optics_labels = optics.fit_predict(X)
    
    # 4. BorderPeeling
    border_peeling = BorderPeeling()
    border_peeling.fit(X)
    border_labels = border_peeling.labels_
    
    # 5. AUTOCLUST
    autoclust = AUTOCLUST()
    autoclust.fit(X)
    autoclust_labels = autoclust.labels_
    
    # 创建子图布局
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3)
    
    # 1. 真实聚类
    ax1 = plt.subplot(gs[0, 0])
    ax1.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis')
    ax1.set_title('真实聚类')
    
    # 2. 桥点聚类
    ax2 = plt.subplot(gs[0, 1])
    ax2.scatter(X[:, 0], X[:, 1], c=bridge_labels, cmap='viridis')
    ax2.set_title('桥点聚类')
    
    # 3. HDBSCAN
    ax3 = plt.subplot(gs[0, 2])
    ax3.scatter(X[:, 0], X[:, 1], c=hdbscan_labels, cmap='viridis')
    ax3.set_title('HDBSCAN')
    
    # 4. OPTICS
    ax4 = plt.subplot(gs[1, 0])
    ax4.scatter(X[:, 0], X[:, 1], c=optics_labels, cmap='viridis')
    ax4.set_title('OPTICS')
    
    # 5. BorderPeeling
    ax5 = plt.subplot(gs[1, 1])
    ax5.scatter(X[:, 0], X[:, 1], c=border_labels, cmap='viridis')
    ax5.set_title('BorderPeeling')
    
    # 6. AUTOCLUST
    ax6 = plt.subplot(gs[1, 2])
    ax6.scatter(X[:, 0], X[:, 1], c=autoclust_labels, cmap='viridis')
    ax6.set_title('AUTOCLUST')
    
    # 确保所有子图都显示完整的边框
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
    
    # 设置整体标题
    plt.suptitle(f'多种聚类算法结果比较 - {filename}', fontsize=16)
    # 调整布局，为整体标题留出空间，增加右侧边距确保右边框可见
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])

    return fig


def plot_multi_clustering_with_tsne(filename, X, true_labels):
    """
    使用TSNE降维后在同一张图上展示多种聚类算法的结果
    适用于高维数据集
    
    参数:
    filename: 数据集文件名
    X: 数据集特征
    true_labels: 真实的聚类标签
    """
    # 使用TSNE进行降维
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # 设置不同的聚类算法
    # 1. 桥点聚类 (Bridge Clustering)
    kls = LocalOutlierFactor
    args = {'contamination': .2, 'n_neighbors': 15}
    k = 7
    od = kls(**args)
    bridge_clf = BridgeClustering(od, k)
    bridge_clf.fit(X)
    bridge_labels = bridge_clf.labels_
    
    # 2. HDBSCAN
    hdbscan_clf = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
    hdbscan_labels = hdbscan_clf.fit_predict(X)
    
    # 3. OPTICS
    # 计算样本数的7%作为min_samples参数
    min_samples = max(int(X.shape[0] * 0.07), 2)  # 确保至少为2
    optics = OPTICS(min_samples=min_samples)
    optics_labels = optics.fit_predict(X)
    
    # 4. BorderPeeling
    border_peeling = BorderPeeling(k=20)
    border_peeling.fit(X)
    border_labels = border_peeling.labels_
    
    # 5. AUTOCLUST
    autoclust = AUTOCLUST()
    autoclust.fit(X)
    autoclust_labels = autoclust.labels_
    
    # 创建子图布局
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3)
    
    # 1. 真实聚类
    ax1 = plt.subplot(gs[0, 0])
    ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], c=true_labels, cmap='viridis')
    ax1.set_title('真实聚类')
    
    # 2. 桥点聚类
    ax2 = plt.subplot(gs[0, 1])
    ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=bridge_labels, cmap='viridis')
    ax2.set_title('桥点聚类')
    
    # 3. HDBSCAN
    ax3 = plt.subplot(gs[0, 2])
    ax3.scatter(X_tsne[:, 0], X_tsne[:, 1], c=hdbscan_labels, cmap='viridis')
    ax3.set_title('HDBSCAN')
    
    # 4. OPTICS
    ax4 = plt.subplot(gs[1, 0])
    ax4.scatter(X_tsne[:, 0], X_tsne[:, 1], c=optics_labels, cmap='viridis')
    ax4.set_title('OPTICS')
    
    # 5. BorderPeeling
    ax5 = plt.subplot(gs[1, 1])
    ax5.scatter(X_tsne[:, 0], X_tsne[:, 1], c=border_labels, cmap='viridis')
    ax5.set_title('BorderPeeling')
    
    # 6. AUTOCLUST
    ax6 = plt.subplot(gs[1, 2])
    ax6.scatter(X_tsne[:, 0], X_tsne[:, 1], c=autoclust_labels, cmap='viridis')
    ax6.set_title('AUTOCLUST')
    
    # 确保所有子图都显示完整的边框
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
    
    # 设置整体标题
    plt.suptitle(f'多种聚类算法结果比较 (TSNE降维) - {filename}', fontsize=16)
    # 调整布局，为整体标题留出空间，增加右侧边距确保右边框可见
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])
    
    return fig


if __name__ == '__main__':
    # 数据集路径设置
    data_folder = './data/datasets/'
    artificial_folder = 'synthetic/'
    real_world_folder = 'real-world/'
    select_folder = artificial_folder  # 选择数据集类型
    folder = data_folder + select_folder
    
    # 结果保存路径设置
    save_path = './result/'
    multi_clusters_savepath = save_path + 'multi_clusters/' + select_folder
    # 修改保存配置，增加右侧边距以确保右边框不被裁剪
    plot_config = {'bbox_inches': 'tight', 'pad_inches': 0.1}
    
    # 确保保存目录存在
    if not os.path.exists(multi_clusters_savepath):
        os.makedirs(multi_clusters_savepath)
    
    # 遍历数据集文件夹中的所有文件
    for filename in os.listdir(folder):
        dataPath = folder + filename
        print(f'处理数据集: {filename}')
        
        # 读取数据集
        X, cluster_labels = read_dataset(dataPath)
        
        # 根据数据集类型选择可视化方法
        if select_folder == artificial_folder:
            # 对于人工合成数据集，直接可视化
            cluster_fig = plot_multi_clustering(filename, X, cluster_labels)
        else:
            # 对于真实世界数据集，使用TSNE降维后可视化
            cluster_fig = plot_multi_clustering_with_tsne(filename, X, cluster_labels)
        
        # 保存结果图
        cluster_savepath = f'{multi_clusters_savepath}{filename}.png'
        cluster_fig.savefig(cluster_savepath, **plot_config)
        plt.close(cluster_fig)
        
    print('所有数据集处理完成！')