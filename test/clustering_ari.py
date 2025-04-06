#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
聚类算法性能评估脚本
用于对synthetic和real-world两种数据集进行聚类分析，并计算ARI指标

用法：
    python clustering_ari.py

结果：
    在result目录下生成6个Excel文件，分别保存kmeans、dbscan和hierarchical三种算法
    对synthetic和real-world两种数据集的ARI结果
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

# 获取当前文件所在的目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上级目录的绝对路径
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

from utils.data_utils import read_dataset
from utils.write import read_performance_csv, write_performance_csv

# 数据集路径
data_folder = os.path.join(parent_dir, 'data', 'datasets')
synthetic_folder = os.path.join(data_folder, 'synthetic')
real_world_folder = os.path.join(data_folder, 'real-world')

# 结果保存路径
result_folder = os.path.join(parent_dir, 'result')

# 创建算法对应的文件夹
kmeans_folder = os.path.join(result_folder, 'kmeans')
dbscan_folder = os.path.join(result_folder, 'dbscan')
hierarchical_folder = os.path.join(result_folder, 'hierarchical')

# 确保文件夹存在
for folder in [kmeans_folder, dbscan_folder, hierarchical_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)


def preprocess_data(data):
    """
    数据预处理：标准化
    """
    # 标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data


def kmeans_clustering(data, n_clusters=3):
    """
    K-Means聚类
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels


def dbscan_clustering(data, eps=0.5, min_samples=5):
    """
    DBSCAN密度聚类
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    return labels


def hierarchical_clustering(data, n_clusters=3, linkage='ward'):
    """
    层次聚类
    """
    try:
        hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = hc.fit_predict(data)
        return labels
    except Exception as e:
        print(f"层次聚类失败: {str(e)}")
        # 返回所有点为一个簇的标签，以避免程序崩溃
        return np.zeros(data.shape[0], dtype=int)


def evaluate_clustering(data_folder, algorithm_name, clustering_func, params=None):
    """
    评估聚类算法在指定数据集上的性能
    """
    if params is None:
        params = {}
    
    results = {}
    
    for filename in os.listdir(data_folder):
        try:
            # 读取数据集
            data_path = os.path.join(data_folder, filename)
            X, true_labels = read_dataset(data_path)
            
            # 数据预处理
            processed_data = preprocess_data(X)
            
            # 执行聚类
            if algorithm_name == 'kmeans':
                n_clusters = len(np.unique(true_labels))
                predicted_labels = clustering_func(processed_data, n_clusters=n_clusters)
            elif algorithm_name == 'dbscan':
                # 对于DBSCAN，我们使用默认参数，因为它不需要指定聚类数量
                predicted_labels = clustering_func(processed_data, **params)
            elif algorithm_name == 'hierarchical':
                n_clusters = len(np.unique(true_labels))
                predicted_labels = clustering_func(processed_data, n_clusters=n_clusters)
            
            # 计算ARI
            ari = adjusted_rand_score(true_labels, predicted_labels)
            results[filename] = ari
            print(f"{filename}: ARI = {ari:.4f}")
            
        except Exception as e:
            print(f"处理{filename}时出错: {str(e)}")
            results[filename] = np.nan
    
    return results


def save_results(results, output_path):
    """
    将结果保存到Excel文件
    """
    # 创建DataFrame
    df = pd.DataFrame(list(results.items()), columns=['Dataset', 'ARI'])
    df.set_index('Dataset', inplace=True)
    
    # 保存到Excel
    df.to_excel(output_path)
    print(f"结果已保存至: {output_path}")


def main():
    # 定义算法和参数
    algorithms = {
        'kmeans': (kmeans_clustering, {}),
        'dbscan': (dbscan_clustering, {'eps': 0.5, 'min_samples': 5}),
        'hierarchical': (hierarchical_clustering, {})
    }
    
    # 定义数据集
    datasets = {
        'synthetic': synthetic_folder,
        'real-world': real_world_folder
    }
    
    # 对每种算法和每种数据集进行评估
    for algorithm_name, (clustering_func, params) in algorithms.items():
        print(f"\n评估{algorithm_name}算法...")
        
        for dataset_name, dataset_folder in datasets.items():
            print(f"\n数据集: {dataset_name}")
            
            # 评估聚类算法
            results = evaluate_clustering(dataset_folder, algorithm_name, clustering_func, params)
            
            # 保存结果
            algorithm_folder = os.path.join(result_folder, algorithm_name)
            output_path = os.path.join(algorithm_folder, f"{dataset_name}_ari.xlsx")
            save_results(results, output_path)


if __name__ == "__main__":
    main()