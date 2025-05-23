'''
Author       : Leaf(2064944038@qq.com)
Version      : V1.0
Date         : 2025-01-11 17:27:07
Description  : 
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


def plot(filename, X, cluster_labels, predicted_labels):
    # 计算 ARI
    ari = adjusted_rand_score(cluster_labels, predicted_labels)

    # 创建一个网格规范 (GridSpec)，用于更精确地控制布局
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.2])  # 第三个列是窄的，用来放置ARI文本

    # 可视化结果
    fig = plt.figure(figsize=(12, 5))  # 增加宽度以适应额外的文本空间

    # 绘制真实聚类
    ax_true = plt.subplot(gs[0])
    scatter_true = ax_true.scatter(X[:, 0],
                                   X[:, 1],
                                   c=cluster_labels,
                                   cmap='viridis')
    ax_true.set_title('True Clustering')

    # 绘制预测聚类
    ax_pred = plt.subplot(gs[1])
    scatter_pred = ax_pred.scatter(X[:, 0],
                                   X[:, 1],
                                   c=predicted_labels,
                                   cmap='viridis')
    ax_pred.set_title(f'{filename}')

    # 在第三个列中添加 ARI 值到右侧空白处
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(0.92,
             0.5,
             f'ARI: {ari:.4f}',
             transform=fig.transFigure,
             fontsize=12,
             verticalalignment='center',
             horizontalalignment='center',
             bbox=props)

    plt.tight_layout()
    plt.show()
    # plt.close(fig)
    return fig


tsne = TSNE()


def plotWithTSNE(filename, X, cluster_labels, predicted_labels):
    # 计算 ARI
    ari = adjusted_rand_score(cluster_labels, predicted_labels)

    # 对数据进行降维
    tsne.fit_transform(X)
    # 创建 DataFrame 来保存 t-SNE 结果
    # 如果 X 是 numpy 数组，则不使用 X.index
    # 如果 X 是 pandas DataFrame，则可以使用 X.index
    if isinstance(X, pd.DataFrame):
        tsne_data = pd.DataFrame(tsne.embedding_,
                                 index=X.index,
                                 columns=['tsne-one', 'tsne-two'])
    else:
        tsne_data = pd.DataFrame(tsne.embedding_,
                                 columns=['tsne-one', 'tsne-two'])

    # 创建一个网格规范 (GridSpec)，用于更精确地控制布局
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.2])  # 第三个列是窄的，用来放置ARI文本
    # 可视化结果
    fig = plt.figure(figsize=(12, 5))  # 增加宽度以适应额外的文本空间

    # 绘制真实聚类
    ax_true = plt.subplot(gs[0])
    scatter_true = ax_true.scatter(tsne_data['tsne-one'],
                                   tsne_data['tsne-two'],
                                   c=cluster_labels,
                                   cmap='viridis')
    ax_true.set_title('True Clustering')

    # 绘制预测聚类
    ax_pred = plt.subplot(gs[1])
    scatter_pred = ax_pred.scatter(tsne_data['tsne-one'],
                                   tsne_data['tsne-two'],
                                   c=predicted_labels,
                                   cmap='viridis')
    ax_pred.set_title(f'{filename}')

    # 在第三个列中添加 ARI 值到右侧空白处
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(0.92,
             0.5,
             f'ARI: {ari:.4f}',
             transform=fig.transFigure,
             fontsize=12,
             verticalalignment='center',
             horizontalalignment='center',
             bbox=props)

    plt.tight_layout()
    # plt.show()
    plt.close(fig)
    return fig


'''
1 选择数据集
2 选择是否生成性能指标的csv文件 =》 选择指标
3 展示图片还是直接生成
'''
if __name__ == '__main__':
    data_folder = './data/datasets/'
    artificial_folder = 'synthetic/'
    real_world_folder = 'real-world/'
    select_folder = artificial_folder  # f1. 选择数据集
    folder = data_folder + select_folder

    save_path = './result/'
    clusters_savepath = save_path + 'clusters/' + select_folder
    plot_config = {'bbox_inches': 'tight', 'pad_inches': 0}

    update_performance = True  # f2. 是否生成性能指标的csv文件
    if update_performance == True:
        ari_file = 'ari.csv'
        ari_fun = adjusted_rand_score

        select_performance_folder = ari_file  # f2.2. 选择使用的指标
        compute_performance = ari_fun
        performance_path = save_path + 'performance/' + select_folder + select_performance_folder
        performance_df = read_performance_csv(performance_path)

    kls = LocalOutlierFactor
    args = {'contamination': .2, 'n_neighbors': 15}
    k = 7

    for filename in os.listdir(folder):
        # for i in range(0, 1):
        # filename = 'diamond9.arff'
        dataPath = folder + filename
        print(filename)
        X, cluster_labels = read_dataset(dataPath)

        # BAC
        print('BAC')
        od = kls(**args)
        clf = BridgeClustering(od, k)
        clf.fit(X)
        predicted_labels = clf.labels_
        outliers = clf.outliers_

        if update_performance == True:
            performance = compute_performance(cluster_labels, predicted_labels)
            if select_performance_folder == ari_file:
                UpdateARI(performance_df, filename, 'BAC', performance)
        
        # DBSCAN: minpoints=5
        print('DBSCAN')
        dbscan = DBSCAN(min_samples=5)
        dbscan_labels = dbscan.fit_predict(X)
        
        if update_performance == True:
            dbscan_performance = compute_performance(cluster_labels, dbscan_labels)
            if select_performance_folder == ari_file:
                UpdateARI(performance_df, filename, 'DBSCAN', dbscan_performance)
        
        # OPTICS: min-points=7%
        # 计算样本数的7%作为min_samples参数
        # min_samples = max(int(X.shape[0] * 0.07), 2)  # 确保至少为2
        print('OPTICS')
        optics = OPTICS(min_samples=0.07, cluster_method='xi')
        optics_labels = optics.fit_predict(X)
        
        if update_performance == True:
            optics_performance = compute_performance(cluster_labels, optics_labels)
            if select_performance_folder == ari_file:
                UpdateARI(performance_df, filename, 'OPTICS', optics_performance)
        
        # HDBSCAN: min-cluster-size=15, min-points=5
        print('HDBSCAN')
        hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
        hdbscan_labels = hdbscan_clusterer.fit_predict(X)
        
        if update_performance == True:
            hdbscan_performance = compute_performance(cluster_labels, hdbscan_labels)
            if select_performance_folder == ari_file:
                UpdateARI(performance_df, filename, 'HDBSCAN', hdbscan_performance)
        
        # BorderPeeling: k=20
        print('BorderPeeling')
        border_peeling = BorderPeeling(k=20)
        border_peeling.fit(X)
        border_peeling_labels = border_peeling.labels_
        
        if update_performance == True:
            border_peeling_performance = compute_performance(cluster_labels, border_peeling_labels)
            if select_performance_folder == ari_file:
                UpdateARI(performance_df, filename, 'BorderPeeling', border_peeling_performance)
        
        # AUTOCLUST
        print('AUTOCLUST')
        autoclust = AUTOCLUST()
        autoclust.fit(X)
        autoclust_labels = autoclust.labels_
        
        if update_performance == True:
            autoclust_performance = compute_performance(cluster_labels, autoclust_labels)
            if select_performance_folder == ari_file:
                UpdateARI(performance_df, filename, 'AUTOCLUST', autoclust_performance)
        

        # if select_folder == artificial_folder:
        #     cluster_fig = plot(filename, X, cluster_labels, dbscan_labels)
        # else:
        #     cluster_fig = plotWithTSNE(filename, X, cluster_labels,
        #                                dbscan_labels)

        # cluster_savepath = f'{clusters_savepath}{filename}.png'
        # cluster_fig.savefig(cluster_savepath, **plot_config)

    if update_performance == True:
        write_performance_csv(performance_df, performance_path)
