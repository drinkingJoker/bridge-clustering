'''
Author       : Leaf(2064944038@qq.com)
Version      : V1.0
Date         : 2025-01-11 17:27:07
Description  : 将多种聚类算法的结果在同一张图片上进行对比展示
'''
import os
import sys

from sklearn.metrics import adjusted_rand_score

# 获取当前文件所在的目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上级目录的绝对路径
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

import pandas as pd
from matplotlib import gridspec, pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, OPTICS
import hdbscan
import numpy as np
from scipy.spatial.distance import pdist, squareform

from bridge_clustering import BridgeClustering
from utils.border_peel.border_peeling import BorderPeel as BorderPeeling
from utils.autoclust import AUTOCLUST

from utils.function import compute_neighbors, determine_bridges
from utils.data_utils import read_arff, read_dataset
from utils.write import UpdateARI, read_performance_csv, write_performance_csv


def plot_multiple_clustering(filename, X, cluster_labels, algorithm_labels_dict):
    """
    将多种聚类算法的结果在同一张图片上进行对比展示
    
    参数:
        filename: 数据集文件名
        X: 数据集特征矩阵
        cluster_labels: 真实的聚类标签
        algorithm_labels_dict: 字典，键为算法名称，值为该算法的聚类结果标签
    
    返回:
        fig: matplotlib图像对象
    """
    # 计算算法数量
    n_algorithms = len(algorithm_labels_dict)
    
    # 创建一个网格规范 (GridSpec)，用于更精确地控制布局
    # 第一行放置真实聚类结果，第二行和第三行放置算法聚类结果
    n_rows = (n_algorithms + 1) // 3 + 1  # 每行最多放3个子图
    n_cols = min(3, n_algorithms + 1)     # 最多3列
    
    # 创建图像
    fig = plt.figure(figsize=(15, 5 * n_rows))  # 根据行数调整高度
    gs = gridspec.GridSpec(n_rows, n_cols)
    
    # 绘制真实聚类
    ax_true = plt.subplot(gs[0, 0])
    scatter_true = ax_true.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
    ax_true.set_title('True Clustering')
    
    # 绘制各算法的聚类结果
    for i, (algorithm_name, predicted_labels) in enumerate(algorithm_labels_dict.items()):
        # 计算当前子图的行和列位置
        row = (i + 1) // n_cols
        col = (i + 1) % n_cols
        
        # 创建子图
        ax = plt.subplot(gs[row, col])
        
        # 绘制聚类结果
        scatter = ax.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis')
        
        # 计算ARI
        ari = adjusted_rand_score(cluster_labels, predicted_labels)
        
        # 设置标题，包含算法名称和ARI值
        ax.set_title(f'{algorithm_name} (ARI: {ari:.4f})')
    
    plt.tight_layout()
    return fig


def plot_multiple_clustering_tsne(filename, X, cluster_labels, algorithm_labels_dict):
    """
    使用t-SNE降维后，将多种聚类算法的结果在同一张图片上进行对比展示
    
    参数:
        filename: 数据集文件名
        X: 数据集特征矩阵
        cluster_labels: 真实的聚类标签
        algorithm_labels_dict: 字典，键为算法名称，值为该算法的聚类结果标签
    
    返回:
        fig: matplotlib图像对象
    """
    # 对数据进行降维
    tsne = TSNE()
    tsne.fit_transform(X)
    
    # 创建 DataFrame 来保存 t-SNE 结果
    if isinstance(X, pd.DataFrame):
        tsne_data = pd.DataFrame(tsne.embedding_, index=X.index, columns=['tsne-one', 'tsne-two'])
    else:
        tsne_data = pd.DataFrame(tsne.embedding_, columns=['tsne-one', 'tsne-two'])
    
    # 计算算法数量
    n_algorithms = len(algorithm_labels_dict)
    
    # 创建一个网格规范 (GridSpec)，用于更精确地控制布局
    # 第一行放置真实聚类结果，第二行和第三行放置算法聚类结果
    n_rows = (n_algorithms + 1) // 3 + 1  # 每行最多放3个子图
    n_cols = min(3, n_algorithms + 1)     # 最多3列
    
    # 创建图像
    fig = plt.figure(figsize=(15, 5 * n_rows))  # 根据行数调整高度
    gs = gridspec.GridSpec(n_rows, n_cols)
    
    # 绘制真实聚类
    ax_true = plt.subplot(gs[0, 0])
    scatter_true = ax_true.scatter(tsne_data['tsne-one'], tsne_data['tsne-two'], c=cluster_labels, cmap='viridis')
    ax_true.set_title('True Clustering')
    
    # 绘制各算法的聚类结果
    for i, (algorithm_name, predicted_labels) in enumerate(algorithm_labels_dict.items()):
        # 计算当前子图的行和列位置
        row = (i + 1) // n_cols
        col = (i + 1) % n_cols
        
        # 创建子图
        ax = plt.subplot(gs[row, col])
        
        # 绘制聚类结果
        scatter = ax.scatter(tsne_data['tsne-one'], tsne_data['tsne-two'], c=predicted_labels, cmap='viridis')
        
        # 计算ARI
        ari = adjusted_rand_score(cluster_labels, predicted_labels)
        
        # 设置标题，包含算法名称和ARI值
        ax.set_title(f'{algorithm_name} (ARI: {ari:.4f})')
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    data_folder = './data/datasets/'
    artificial_folder = 'synthetic/'
    real_world_folder = 'real-world/'
    select_folder = artificial_folder  # 选择数据集
    folder = data_folder + select_folder

    save_path = './result/'
    clusters_savepath = save_path + 'compare_clusters/' + select_folder
    
    # 确保保存路径存在
    os.makedirs(clusters_savepath, exist_ok=True)
    
    plot_config = {'bbox_inches': 'tight', 'pad_inches': 0}

    update_performance = True  # 是否生成性能指标的csv文件
    if update_performance == True:
        ari_file = 'ari.csv'
        ari_fun = adjusted_rand_score

        select_performance_folder = ari_file  # 选择使用的指标
        compute_performance = ari_fun
        performance_path = save_path + 'performance/' + select_folder + select_performance_folder
        performance_df = read_performance_csv(performance_path)

    kls = LocalOutlierFactor
    args = {'contamination': .2, 'n_neighbors': 15}
    k = 7

    for filename in os.listdir(folder):
        dataPath = folder + filename
        print(f"处理数据集: {filename}")
        X, cluster_labels = read_dataset(dataPath)
        
        # 存储各算法的聚类结果
        algorithm_labels_dict = {}
        
        # BAC (桥点聚类)
        print('BAC')
        od = kls(**args)
        clf = BridgeClustering(od, k)
        clf.fit(X)
        bac_labels = clf.labels_
        algorithm_labels_dict['BAC'] = bac_labels
        
        if update_performance == True:
            bac_performance = compute_performance(cluster_labels, bac_labels)
            if select_performance_folder == ari_file:
                UpdateARI(performance_df, filename, 'BAC', bac_performance)
        
        # DBSCAN: minpoints=5
        print('DBSCAN')
        dbscan = DBSCAN(min_samples=5)
        dbscan_labels = dbscan.fit_predict(X)
        algorithm_labels_dict['DBSCAN'] = dbscan_labels
        
        if update_performance == True:
            dbscan_performance = compute_performance(cluster_labels, dbscan_labels)
            if select_performance_folder == ari_file:
                UpdateARI(performance_df, filename, 'DBSCAN', dbscan_performance)
        
        # OPTICS: min-points=7%
        print('OPTICS')
        optics = OPTICS(min_samples=0.07, cluster_method='xi')
        optics_labels = optics.fit_predict(X)
        algorithm_labels_dict['OPTICS'] = optics_labels
        
        if update_performance == True:
            optics_performance = compute_performance(cluster_labels, optics_labels)
            if select_performance_folder == ari_file:
                UpdateARI(performance_df, filename, 'OPTICS', optics_performance)
        
        # HDBSCAN: min-cluster-size=15, min-points=5
        print('HDBSCAN')
        hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
        hdbscan_labels = hdbscan_clusterer.fit_predict(X)
        algorithm_labels_dict['HDBSCAN'] = hdbscan_labels
        
        if update_performance == True:
            hdbscan_performance = compute_performance(cluster_labels, hdbscan_labels)
            if select_performance_folder == ari_file:
                UpdateARI(performance_df, filename, 'HDBSCAN', hdbscan_performance)
        
        # BorderPeeling: k=20
        print('BorderPeeling')
        border_peeling = BorderPeeling(k=20)
        border_peeling.fit(X)
        border_peeling_labels = border_peeling.labels_
        algorithm_labels_dict['BorderPeeling'] = border_peeling_labels
        
        if update_performance == True:
            border_peeling_performance = compute_performance(cluster_labels, border_peeling_labels)
            if select_performance_folder == ari_file:
                UpdateARI(performance_df, filename, 'BorderPeeling', border_peeling_performance)
        
        # AUTOCLUST
        print('AUTOCLUST')
        autoclust = AUTOCLUST()
        autoclust.fit(X)
        autoclust_labels = autoclust.labels_
        algorithm_labels_dict['AUTOCLUST'] = autoclust_labels
        
        if update_performance == True:
            autoclust_performance = compute_performance(cluster_labels, autoclust_labels)
            if select_performance_folder == ari_file:
                UpdateARI(performance_df, filename, 'AUTOCLUST', autoclust_performance)
        
        # 绘制并保存图像
        if select_folder == artificial_folder:
            # 对于人工数据集，直接绘制
            cluster_fig = plot_multiple_clustering(filename, X, cluster_labels, algorithm_labels_dict)
        else:
            # 对于真实世界数据集，使用t-SNE降维后绘制
            cluster_fig = plot_multiple_clustering_tsne(filename, X, cluster_labels, algorithm_labels_dict)
        
        # 保存图像
        cluster_savepath = f'{clusters_savepath}{filename}.png'
        cluster_fig.savefig(cluster_savepath, **plot_config)
        plt.close(cluster_fig)

    if update_performance == True:
        write_performance_csv(performance_df, performance_path)