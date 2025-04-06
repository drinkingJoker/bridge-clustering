#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
桥点聚类实现
用于图像分割的桥点聚类算法
"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


def predict_and_cluster(clf, X, k, n_clusters=None):
    """ 进行异常点检测并执行聚类 """
    # 计算哪些点是异常点（桥接点）
    pred_bridges = clf.fit_predict(X) == -1
    
    # 计算k近邻
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(X)
    nn_distances, nn_indices = nn.kneighbors(X)
    
    # 移除自连接
    nn_distances = nn_distances[:, 1:]
    nn_indices = nn_indices[:, 1:]
    
    # 计算聚类标签
    labels = compute_cluster_labels(X, k, pred_bridges, nn_indices, nn_distances)
    
    # 确保聚类类别数满足 n_clusters
    if n_clusters is not None:
        unique_labels = np.unique(labels)
        if len(unique_labels) > n_clusters:
            print(f"当前聚类数 {len(unique_labels)} 超过目标 {n_clusters}，进行合并...")
            labels = merge_clusters(labels, n_clusters)
    
    return labels, pred_bridges


def compute_cluster_labels(X, k, is_bridge_candidate, local_indices, local_distances):
    """ 计算聚类标签 """
    # 扩展非桥点的标签
    pred_labels = expand_labels(local_indices, is_bridge_candidate)
    
    # 为桥点分配标签
    pred_labels = assign_bridge_labels(pred_labels, local_indices, local_distances)
    
    return pred_labels


def expand_labels(local_indices, is_bridge_candidate):
    """ 扩展非桥点的标签 """
    n_points = local_indices.shape[0]
    cluster_labels = np.full(n_points, -1, dtype=int)
    
    # 构建邻接表
    from collections import defaultdict
    adj_list = defaultdict(list)
    for i in range(n_points):
        if is_bridge_candidate[i]:
            continue  # 桥接点不参与连通性计算
        for neighbor in local_indices[i]:
            if not is_bridge_candidate[neighbor] and neighbor != i:
                adj_list[i].append(neighbor)
                adj_list[neighbor].append(i)  # 确保双向连接
    
    # 进行连通分量标记
    label = 0
    visited = np.zeros(n_points, dtype=bool)
    
    def dfs(node):
        stack = [node]
        while stack:
            el = stack.pop()
            if visited[el]:
                continue
            visited[el] = True
            cluster_labels[el] = label
            stack.extend(adj_list[el])  # 继续遍历邻接点
    
    for i in range(n_points):
        if is_bridge_candidate[i] or visited[i]:
            continue
        dfs(i)
        label += 1
    
    return cluster_labels


def assign_bridge_labels(cluster_labels, local_indices, local_distances):
    """ 为桥点分配标签 """
    import heapq
    from collections import defaultdict
    
    # 生成邻接表
    n_points = local_indices.shape[0]
    adj_list = defaultdict(list)
    
    for i in range(n_points):
        for j, dist in zip(local_indices[i], local_distances[i]):
            if i != j:  # 避免自环
                adj_list[i].append((dist, j))  # (距离, 目标点)
                adj_list[j].append((dist, i))  # 确保双向连接
    
    is_bridge_candidate = cluster_labels == -1
    missing_indexes = np.where(is_bridge_candidate)[0]
    
    # 优先队列 (min-heap)
    pq = []
    # 存储未分配点的邻居信息
    pending_entries = defaultdict(list)
    
    # 初始化优先队列
    for i in missing_indexes:
        for dist, neigh in adj_list[i]:
            if cluster_labels[neigh] != -1:
                heapq.heappush(pq, (dist, neigh, i))
            else:
                # 记录未标记邻居的边，等待邻居被处理时处理
                pending_entries[neigh].append((dist, i))
    
    # 处理桥接点
    while pq:
        dist, src, cur = heapq.heappop(pq)
        
        if cluster_labels[cur] != -1:
            continue  # 已处理
        
        cluster_labels[cur] = cluster_labels[src]
        
        # 处理当前点未处理的边
        if cur in pending_entries:
            for pending_dist, pending_cur in pending_entries[cur]:
                if cluster_labels[pending_cur] == -1:
                    heapq.heappush(pq, (pending_dist, cur, pending_cur))
            del pending_entries[cur]  # 移除已处理
    
    # 处理剩余未分配的点
    remaining_missing = np.where(cluster_labels == -1)[0]
    for i in remaining_missing:
        min_dist = float('inf')
        best_label = -1
        for dist, neigh in adj_list[i]:
            if cluster_labels[neigh] != -1 and dist < min_dist:
                min_dist = dist
                best_label = cluster_labels[neigh]
        if best_label != -1:
            cluster_labels[i] = best_label
        else:
            # 极端情况：所有邻居均为桥接点，强制标记
            cluster_labels[i] = 0
    
    return cluster_labels


def merge_clusters(labels, n_clusters):
    """ 如果聚类类别数超过 n_clusters，则合并 """
    unique_labels = np.unique(labels)
    if len(unique_labels) <= n_clusters:
        return labels
    
    # 使用 KMeans 进行聚类合并
    from sklearn.cluster import KMeans
    new_labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(labels.reshape(-1, 1))
    
    return new_labels


class BridgeClustering:
    def __init__(self, outlier_detection, k, n_clusters=None):
        """
        :param outlier_detection: 用于桥接点检测的异常检测模型
        :param k: 近邻数
        :param n_clusters: 目标聚类数
        """
        self.outlier_detection_ = outlier_detection
        self.k_ = k
        self.n_clusters_ = n_clusters
    
    def fit(self, X, y=None):
        """ 执行聚类 """
        self.labels_, self.outliers_ = predict_and_cluster(
            self.outlier_detection_, X, self.k_, self.n_clusters_)
        return self
    
    def fit_predict(self, X, y=None):
        """ 训练并返回聚类标签 """
        self.fit(X)
        return self.labels_