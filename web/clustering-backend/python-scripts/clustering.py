#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
聚类分析脚本
用于对数据集进行聚类分析并生成可视化结果

用法：
    python clustering.py <input_file> <output_file> <algorithm> [--param value]

参数：
    input_file: 输入数据集文件路径（CSV或TXT格式）
    output_file: 输出图像文件路径
    algorithm: 聚类算法（kmeans, dbscan, hierarchical, bridge）
    --param value: 算法特定参数
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from arff_loader import load_arff_file
import sys
import os
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
# 将项目根目录添加到系统路径
if project_root not in sys.path:
    sys.path.append(project_root)
from bridge_clustering import BridgeClustering


def load_dataset(file_path):
    """
    加载数据集文件
    支持CSV、TXT和ARFF格式
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext == '.csv':
            data = pd.read_csv(file_path)
        elif file_ext == '.txt':
            data = pd.read_csv(file_path, sep='\t')
        elif file_ext == '.arff':
            data = load_arff_file(file_path)
        else:
            # 尝试自动检测分隔符
            data = pd.read_csv(file_path, sep=None, engine='python')

        # 移除可能的ID列或非数值列
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < data.shape[1]:
            print(f"警告：移除了{data.shape[1] - len(numeric_cols)}个非数值列")
            data = data[numeric_cols]

        return data
    except Exception as e:
        print(f"加载数据集失败: {str(e)}")
        sys.exit(1)


def preprocess_data(data):
    """
    数据预处理：标准化和降维（如果维度过高）
    """
    # 标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # 如果维度大于2，使用PCA降维以便可视化
    if scaled_data.shape[1] > 2:
        pca = PCA(n_components=2)
        return pca.fit_transform(scaled_data)

    return scaled_data


def kmeans_clustering(data, params):
    """
    K-Means聚类
    """
    k = int(params.get('k', 3)) if params.get('k') is not None else 3
    max_iter = int(params.get(
        'maxIter', 300)) if params.get('maxIter') is not None else 300
    init = params.get(
        'init', 'k-means++') if params.get('init') is not None else 'k-means++'

    kmeans = KMeans(n_clusters=k,
                    init=init,
                    max_iter=max_iter,
                    random_state=42)
    labels = kmeans.fit_predict(data)

    # 计算轮廓系数
    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(data, labels)
        print(f"轮廓系数: {silhouette:.3f}")

    return labels, kmeans.cluster_centers_


def dbscan_clustering(data, params):
    """
    DBSCAN密度聚类
    """
    eps = float(params.get('eps', 0.5))
    min_pts = int(params.get('minPts', 5))
    metric = params.get('metric', 'euclidean')

    dbscan = DBSCAN(eps=eps, min_samples=min_pts, metric=metric)
    labels = dbscan.fit_predict(data)

    # 计算轮廓系数（如果不是所有点都是噪声）
    if len(np.unique(labels)) > 1 and -1 not in labels:
        silhouette = silhouette_score(data, labels)
        print(f"轮廓系数: {silhouette:.3f}")

    # DBSCAN没有簇中心，返回None
    return labels, None


def hierarchical_clustering(data, params):
    """
    层次聚类
    """
    try:
        # 确保参数类型正确
        try:
            n_clusters = int(params.get('clusters', 3))
            if n_clusters <= 0:
                print(f"警告：聚类数量必须大于0，当前值为{n_clusters}，已自动调整为2")
                n_clusters = 2
        except (ValueError, TypeError) as e:
            print(f"警告：聚类数量参数无效，使用默认值3: {str(e)}")
            n_clusters = 3

        linkage = params.get('linkage', 'ward')
        # 验证linkage参数
        valid_linkages = ['ward', 'complete', 'average', 'single']
        if linkage not in valid_linkages:
            print(f"警告：无效的连接方式 '{linkage}'，已自动调整为'ward'")
            linkage = 'ward'

        metric = params.get('metric', 'euclidean')
        # 验证metric参数
        valid_metrics = ['euclidean', 'manhattan', 'cosine', 'l1', 'l2']
        if metric not in valid_metrics:
            print(f"警告：无效的距离度量 '{metric}'，已自动调整为'euclidean'")
            metric = 'euclidean'

        # 注意：ward连接只能使用欧氏距离
        if linkage == 'ward' and metric != 'euclidean':
            print("警告：Ward连接只能使用欧氏距离，已自动调整")
            metric = 'euclidean'

        # 检查数据是否为空或包含NaN值
        if data.shape[0] == 0:
            raise ValueError("数据集为空，无法进行聚类")
        if np.isnan(data).any():
            print("警告：数据集包含NaN值，已自动移除")
            data = data[~np.isnan(data).any(axis=1)]
            if data.shape[0] == 0:
                raise ValueError("移除NaN值后数据集为空，无法进行聚类")

        print(
            f"执行层次聚类: n_clusters={n_clusters}, linkage={linkage}, metric={metric}"
        )
        # 根据linkage类型决定参数传递方式
        if linkage == 'ward':
            # ward连接只接受欧氏距离，不需要额外指定
            hc = AgglomerativeClustering(n_clusters=n_clusters,
                                         linkage=linkage)
        else:
            # 非ward连接需要使用affinity参数而不是metric参数
            hc = AgglomerativeClustering(n_clusters=n_clusters,
                                         linkage=linkage,
                                         affinity=metric)
        labels = hc.fit_predict(data)

        # 计算轮廓系数
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(data, labels)
            print(f"轮廓系数: {silhouette:.3f}")

        # 层次聚类没有簇中心，返回None
        return labels, None

    except Exception as e:
        print(f"层次聚类失败: {str(e)}")
        # 返回所有点为一个簇的标签，以避免程序崩溃
        print("返回默认聚类结果（所有点归为一个簇）")
        return np.zeros(data.shape[0], dtype=int), None


def bridge_clustering(data, params):
    """
    基于桥点的密度聚类
    使用已实现的BridgeClustering类
    """

    # 获取参数

    n_clusters = int(params.get(
        'clusters', None)) if params.get('clusters') is not None else None
    contamination = float(
        params.get('contamination',
                   0.2)) if params.get('contamination') is not None else 0.2
    n_neighbors_od = int(
        params.get('n_neighbors_od',
                   15)) if params.get('n_neighbors_od') is not None else 15
    n_neighbors = int(params.get(
        'n_neighbors', 7)) if params.get('n_neighbors') is not None else 7
    od = params.get(
        'outlierDetection',
        'LOF') if params.get('outlierDetection') is not None else 'LOF'
    if od == 'LOF':
        od = LocalOutlierFactor(n_neighbors=n_neighbors_od,
                                contamination=contamination)
    elif od == 'IF':
        od = IsolationForest(contamination=contamination,
                              random_state=42)

    # 创建BridgeClustering实例并执行聚类
    bridge_clustering = BridgeClustering(od, n_neighbors, n_clusters)
    labels = bridge_clustering.fit_predict(data)

    # 计算轮廓系数（如果不是所有点都是噪声）
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1 and -1 not in unique_labels:
        try:
            silhouette = silhouette_score(data, labels)
            print(f"轮廓系数: {silhouette:.3f}")
        except Exception as e:
            print(f"计算轮廓系数失败: {str(e)}")

    return labels, None


def visualize_clusters(data, labels, centers, output_file):
    """
    可视化聚类结果并保存为图像
    """
    plt.figure(figsize=(10, 8))

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 获取唯一的簇标签
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if -1 in unique_labels:  # 如果有噪声点（-1标签）
        n_clusters -= 1

    # 为每个簇选择不同的颜色
    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters + 1))

    # 绘制数据点
    for i, label in enumerate(unique_labels):
        if label == -1:
            # 噪声点用黑色表示
            cluster_data = data[labels == label]
            plt.scatter(cluster_data[:, 0],
                        cluster_data[:, 1],
                        s=50,
                        c='black',
                        marker='x',
                        alpha=0.5,
                        label='噪声点')
        else:
            # 正常簇用彩色表示
            cluster_data = data[labels == label]
            color_idx = i
            if -1 in unique_labels:
                color_idx = i - 1 if i > 0 else i
            plt.scatter(cluster_data[:, 0],
                        cluster_data[:, 1],
                        s=50,
                        c=[colors[color_idx]],
                        alpha=0.7,
                        label=f'簇 {label+1}')

    # 绘制簇中心（如果有）
    if centers is not None:
        plt.scatter(centers[:, 0],
                    centers[:, 1],
                    s=200,
                    c='red',
                    marker='*',
                    alpha=0.7,
                    label='簇中心')

    plt.title('聚类结果可视化')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图像
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"聚类结果已保存至: {output_file}")


def generate_dataset_info(data, processed_data, labels):
    """
    生成数据集信息和样本数据的JSON
    """
    import json
    
    # 获取数据集基本信息
    sample_count, dimensions = data.shape
    
    # 准备样本数据 - 如果样本太多，只展示前三个和后三个
    samples = []
    if sample_count <= 6:
        # 如果样本数量少于等于6，全部展示
        for i in range(sample_count):
            sample_data = data.iloc[i].tolist() if hasattr(data, 'iloc') else data[i].tolist()
            samples.append({
                'index': i,
                'data': sample_data,
                'cluster': int(labels[i])
            })
    else:
        # 如果样本数量大于6，展示前三个和后三个
        for i in range(3):
            sample_data = data.iloc[i].tolist() if hasattr(data, 'iloc') else data[i].tolist()
            samples.append({
                'index': i,
                'data': sample_data,
                'cluster': int(labels[i])
            })
        
        # 添加省略号标记
        samples.append({'index': '...', 'data': '...', 'cluster': '...'})
        
        # 添加后三个样本
        for i in range(sample_count - 3, sample_count):
            sample_data = data.iloc[i].tolist() if hasattr(data, 'iloc') else data[i].tolist()
            samples.append({
                'index': i,
                'data': sample_data,
                'cluster': int(labels[i])
            })
    
    # 创建结果字典
    result = {
        'sampleCount': sample_count,
        'dimensions': dimensions,
        'samples': samples
    }
    
    return result

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='数据集聚类分析')
    parser.add_argument('input_file', help='输入数据集文件路径')
    parser.add_argument('output_file', help='输出图像文件路径')
    parser.add_argument('algorithm',
                        help='聚类算法 (kmeans, dbscan, hierarchical, bridge)')
    parser.add_argument('json_file', help='输出JSON数据文件路径')

    # 添加K-Means参数
    parser.add_argument('--k', type=int, help='聚类数量 (K-Means)')
    parser.add_argument('--maxIter', type=int, help='最大迭代次数 (K-Means)')
    parser.add_argument('--init', help='初始化方法 (K-Means)')

    # 添加DBSCAN参数
    parser.add_argument('--eps', type=float, help='邻域半径 (DBSCAN)')
    parser.add_argument('--minPts', type=int, help='最小样本数 (DBSCAN)')

    # 添加层次聚类参数
    parser.add_argument('--clusters', type=int, help='聚类数量 (层次聚类/桥点聚类)')
    parser.add_argument('--linkage', help='连接方式 (层次聚类)')

    # 添加桥点聚类参数
    parser.add_argument('--outlierDetection', type=str, help='异常检测方法 (桥点聚类)')
    parser.add_argument('--contamination', type=float, help='异常点比例 (桥点聚类)')
    parser.add_argument('--n_neighbors_od',
                        type=int,
                        help='局部异常因子的邻居数量 (桥点聚类)')
    parser.add_argument('--n_neighbors', type=int, help='桥点聚类的邻居数量 (桥点聚类)')

    # 添加通用参数
    parser.add_argument('--metric', help='距离度量')

    # 解析参数
    args = parser.parse_args()

    # 将参数转换为字典
    params = vars(args)

    # 加载数据集
    print(f"加载数据集: {args.input_file}")
    data = load_dataset(args.input_file)
    print(f"数据集大小: {data.shape}")

    # 数据预处理
    print("数据预处理...")
    processed_data = preprocess_data(data)

    # 执行聚类
    print(f"执行{args.algorithm}聚类...")
    if args.algorithm == 'kmeans':
        labels, centers = kmeans_clustering(processed_data, params)
    elif args.algorithm == 'dbscan':
        labels, centers = dbscan_clustering(processed_data, params)
    elif args.algorithm == 'hierarchical':
        labels, centers = hierarchical_clustering(processed_data, params)
    elif args.algorithm == 'bridge':
        labels, centers = bridge_clustering(processed_data, params)
    else:
        print(f"错误：不支持的聚类算法 {args.algorithm}")
        sys.exit(1)

    # 统计聚类结果
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if -1 in unique_labels:  # 如果有噪声点（-1标签）
        n_clusters -= 1

    print(f"聚类结果: {n_clusters}个簇")
    for label in unique_labels:
        if label == -1:
            print(f"  噪声点: {np.sum(labels == label)}个")
        else:
            print(f"  簇 {label+1}: {np.sum(labels == label)}个点")

    # 可视化结果
    print("生成可视化结果...")
    visualize_clusters(processed_data, labels, centers, args.output_file)
    
    # 生成数据集信息和样本数据的JSON文件
    print("生成数据集信息和样本数据...")
    dataset_info = generate_dataset_info(data, processed_data, labels)
    
    # 保存JSON文件
    with open(args.json_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(dataset_info, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
