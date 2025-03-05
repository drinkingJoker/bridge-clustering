# from collections import defaultdict
# import heapq
# import numpy as np
# from sklearn.neighbors import NearestNeighbors, kneighbors_graph, LocalOutlierFactor
# from sklearn.metrics import pairwise_distances
# from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz, find
# from scipy.sparse.csgraph import connected_components
# from sklearn.metrics import adjusted_rand_score
# import matplotlib.pyplot as plt

from utils.function import predict_and_cluster
# def identify_bridge_points(X, k, contamination=0.2):
#     # from sklearn.ensemble import IsolationForest
#     # # 使用Isolation Forest进行异常检测
#     # clf = IsolationForest(contamination=contamination)
#     # clf.fit(X)
#     # outliers = clf.predict(X) == -1

#     # LOF也可以用于异常检测
#     lof = LocalOutlierFactor(contamination=contamination)
#     outliers = lof.fit_predict(X) == -1

#     # plt.scatter(X[outliers, 0], X[outliers, 1])
#     # plt.show()

#     return outliers

# def create_bridge_aware_knn_graph(X, k, bridge_points):
#     # print('create_bridge_aware_knn_graph')
#     # 计算k最近邻图
#     knn_graph = kneighbors_graph(X,
#                                  n_neighbors=k,
#                                  mode='distance',
#                                  include_self=False)
#     # print(knn_graph)

#     # 获取桥点的索引
#     bridge_indices = np.where(bridge_points)[0]
#     # print(bridge_indices)

#     # 创建一个 lil_matrix 来高效地修改稀疏矩阵
#     knn_graph_lil = knn_graph.tolil()

#     # 去除桥点与其它点之间的边，保留桥点与其最近邻点之间的边
#     for i in bridge_indices:
#         neighbors = knn_graph[i].indices
#         distances = knn_graph[i].data
#         # print(f'neighbors: {neighbors}')
#         # print(f'distance: {distances}')
#         min_dis_idx = np.argmin(distances)

#         # for j in neighbors:
#         #     if j == neighbors[min_dis_idx]:
#         #         continue
#         #     knn_graph_lil[i, j] = 0
#         #     knn_graph_lil[j, i] = 0

#         knn_graph_lil[i,:]=0
#         knn_graph_lil[i, neighbors[min_dis_idx]] = distances[min_dis_idx]

#     return knn_graph_lil.tocsr()

# def bridge_aware_clustering(X, k, contamination=0.2):
#     # 识别候选桥点
#     bridge_points = identify_bridge_points(X, k, contamination)

#     # 创建桥点感知的k最近邻图
#     knn_graph = create_bridge_aware_knn_graph(X, k, bridge_points)
#     # print(knn_graph)
#     # 以npz文件格式保存，indices，indptr（i行-（i-1）行得到的是该行的数据数），data
#     # save_npz('log.npz', knn_graph)
#     # 保存稀疏矩阵到文件，以字符串的方式保存
#     # with open('knn_graph.txt', 'w') as file:
#     #     # 写入稀疏矩阵的形状
#     #     file.write(f"{knn_graph.shape[0]} {knn_graph.shape[1]}\n")

#     #     # 使用 find 函数获取非零元素的行索引、列索引和数据
#     #     rows, cols, data = find(knn_graph)

#     #     # 写入非零元素的行、列索引及其值
#     #     for i, j, v in zip(rows, cols, data):
#     #         file.write(f"{i} {j} {v}\n")

#     # 使用连通分量找到簇
#     n_components, labels = connected_components(csgraph=knn_graph,
#                                                 directed=False,
#                                                 return_labels=True)

#     return labels

# def detect_encoding(file_path):
#     with open(file_path, 'rb') as file:
#         raw_data = file.read()
#         result = chardet.detect(raw_data)
#         encoding = result['encoding']
#         print(f"{file_path} Detected encoding: {encoding}")
#         return encoding

# if __name__ == '__main__':
#     # X, true_labels = generate_data()

#     data_folder ='./data/datasets/'
#     artificial_folder = data_folder + 'artificial/'

#     for filename in os.listdir(artificial_folder):
#         print(f'{filename}:')
#         dataPath = artificial_folder + filename
#         # if detect_encoding(dataPath) == False:
#         #     break

#         X, true_labels = Load_arff(dataPath)
#         # print(X, true_labels)

#         # 设置参数
#         k = 10
#         contamination = 0.2

#         # 进行聚类
#         labels = bridge_aware_clustering(X, k, contamination)

#         # 计算调整兰德指数
#         ari = adjusted_rand_score(true_labels, labels)
#         print(f"Adjusted Rand Index: {ari}")

#         # 可视化结果
#         fig, axs = plt.subplots(1,2)
#         fig.figure.set_size_inches(10,5)

#         # print(true_labels)
#         axs[0].scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis')
#         axs[0].set_title('True Clustering')
#         # print(labels)
#         axs[1].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
#         axs[1].set_title(filename)

#         plt.show()


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
