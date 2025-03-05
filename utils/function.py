from collections import defaultdict
import heapq
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time


def remove_self(distances, indices):
    indexes = np.arange(indices.shape[0], dtype=int)[..., np.newaxis]

    mask = indices == indexes
    check_array = (mask.sum(axis=-1)) == 1  # 保证每个样本匹配到了它自己一次
    check = check_array.all()

    if not check:
        print(
            f'WARNING: found at least one point for which the point itself is not included in its neighborhood.'
        )
        print('This is likely due to overlapping points.')
        # 处理那些在 indices 中没有正确匹配到自身的行（即 check_array 为 False 的行），
        # 并强制将这些行的第一个元素设置为匹配自身。这确保了每个样本至少有一个匹配，即使原始数据中存在异常或错误。
        null_elements = ~check_array
        null_indexes = indexes[null_elements]
        mask[null_indexes, 0] = True

    # new_distances = distances[:, 1:] # 不能直接去除第一列然后认为去除了自连接，因为可能存在重合的点，然后该点排在了本点前面！！！
    # new_indices = indices[:, 1:]

    inverse_mask = ~mask

    test_dist = distances[:, 1:]

    new_distances = distances[inverse_mask].reshape(test_dist.shape)
    new_indices = indices[inverse_mask].reshape(test_dist.shape)

    assert (test_dist == new_distances).all()

    return new_distances, new_indices


def compute_neighbors(X, k=None):
    new_k = X.shape[0] if k is None else k + 1
    nn = NearestNeighbors(n_neighbors=new_k, algorithm='ball_tree').fit(X)
    nn_distances, nn_indices = nn.kneighbors(X)
    nn_distances, nn_indices = remove_self(nn_distances, nn_indices)
    return nn_distances, nn_indices


def determine_bridges(labels, local_indices):
    matrix = labels[local_indices]
    self_labels = labels[..., np.newaxis]

    bridges = ~(matrix == self_labels).all(axis=-1)
    return bridges


# def expand_labels(local_indices, is_bridge_candidate):
#     print('expand_labels:')
#     start_time = time.time()  # 记录开始时间

#     n_points = local_indices.shape[0]
#     cluster_labels = np.ones(n_points) * -1  # 初始化所有标签为-1

#     label = 0
#     for i in range(n_points):
#         if is_bridge_candidate[i] or cluster_labels[i] != -1:
#             continue

#         pq = [i]
#         while pq:
#             el = pq.pop()
#             if cluster_labels[el] == -1:  # 只有未标记的元素才需要处理
#                 cluster_labels[el] = label

#                 # 查找当前元素的所有邻居（排除桥接候选点）
#                 neighbors = local_indices[el]

#                 for neighbor in neighbors:
#                     if not is_bridge_candidate[neighbor] and cluster_labels[
#                             neighbor] == -1:
#                         pq.append(neighbor)


#         label += 1
#     # 检查是否所有的非桥点已分配标签
#     non_bridges_unlabeled = np.any((cluster_labels == -1)
#                                    & (~is_bridge_candidate))
#     if non_bridges_unlabeled:
#         print("存在未分配标签的非桥接点。")
#     else:
#         print("所有非桥接点均已成功分配标签。")
#     end_time = time.time()  # 记录结束时间
#     execution_time = end_time - start_time
#     print(f"函数执行时间: {execution_time} 秒")
#     return cluster_labels
def expand_labels(local_indices, is_bridge_candidate):
    print('expand_labels:')
    start_time = time.time()
    n_points = local_indices.shape[0]
    cluster_labels = np.full(n_points, -1, dtype=int)

    # 构建邻接表
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
    non_bridges_unlabeled = np.any(
        np.logical_and(cluster_labels == -1, ~is_bridge_candidate))
    if non_bridges_unlabeled:
        print("存在未分配标签的非桥点。.............................")
    else:
        print("所有非桥点均已成功分配标签。")
    end_time = time.time()  # 记录结束时间
    execution_time = end_time - start_time
    print(f"函数执行时间: {execution_time} 秒")
    return cluster_labels


'''
expand_labels
关键改动点：
移除了 jump_matrix：不再创建一个可能非常大的二维布尔矩阵来存储点与点之间的连接信息。
直接遍历 local_indices：对于每个待处理的点，我们直接检查它的邻居（即 local_indices 中指定的点），并根据这些信息更新聚类标签。
考虑了桥接候选点的影响：在更新聚类标签时，跳过了所有被标记为桥接候选点的点，确保它们不会影响聚类过程。
这种方法大大减少了内存使用，特别是在处理大规模数据集时更为明显。不过需要注意的是，这种优化依赖于 local_indices 的结构以及如何定义邻居关系，确保你的应用场景中可以直接应用这种简化方式。如果 local_indices 包含了复杂的邻接关系或者需要额外的逻辑来确定哪些点是“邻居”，你可能还需要进一步调整此代码。
'''


def generate_adjacency_list(local_distances, local_indices):
    """ 使用邻接表存储点的连接关系，避免 O(n²) 的跳转矩阵 """
    n_points = local_indices.shape[0]
    adj_list = defaultdict(list)

    for i in range(n_points):
        for j, dist in zip(local_indices[i], local_distances[i]):
            if i != j:  # 避免自环
                adj_list[i].append((dist, j))  # (距离, 目标点)
                adj_list[j].append((dist, i))  # 确保双向连接

    return adj_list


def assign_bridge_labels(cluster_labels, local_indices, local_distances):
    """ 采用最小堆 + 邻接表优化 bridge label 计算，并处理未覆盖情况 """
    print('assign_bridge_labels:')
    start_time = time.time()
    adj_list = generate_adjacency_list(local_distances, local_indices)

    n_points = cluster_labels.size
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

    # 二次检查未分配的点（处理环形依赖）
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
            # 极端情况：所有邻居均为桥接点，强制标记（需处理）
            # 此处可选择报错或赋予默认标签（根据需求调整）
            cluster_labels[i] = 0  # 示例：标记为第一个簇
            print(f"警告: 点 {i} 无有效邻居，强制分配标签")

    # 最终检查
    unlabeled = np.where(cluster_labels == -1)[0]
    if unlabeled.size > 0:
        print(f"存在未分配标签的点：{unlabeled.tolist()}")
    else:
        print("所有点均已成功分配标签。")

    print(f"函数执行时间: {time.time() - start_time} 秒")
    return cluster_labels


'''
assign_bridge_labels
关键改动点
移除了对完整跳转矩阵的依赖：我们不再试图创建和操作一个巨大的二维数组来表示所有点之间的关系。
采用了优先队列：通过优先队列（heapq）按距离从小到大处理每个未标记点与其邻居的关系，从而动态地扩展聚类标签。
优化了内存使用：通过直接处理每个点的局部信息（即其邻居的距离和索引），而不是预计算所有可能的连接，大大降低了内存消耗。
这种方法特别适合处理大规模数据集，因为它仅在需要时计算必要的距离，并且以增量方式更新聚类标签。
'''


def compute_cluster_labels(X,
                           k,
                           is_bridge_candidate,
                           local_indices=None,
                           local_distances=None):
    if X is None:
        assert k is None
        assert not local_indices is None
    if local_indices is None:
        local_distances, local_indices = compute_neighbors(X, k)
    assert is_bridge_candidate.dtype == 'bool'

    pred_labels = expand_labels(local_indices, is_bridge_candidate)

    pred_labels = assign_bridge_labels(pred_labels, local_indices,
                                       local_distances)

    return pred_labels


def predict_and_cluster(clf,
                        X,
                        k,
                        n_clusters,
                        local_indices=None,
                        local_distances=None):
    """ 进行异常点检测并执行聚类 """
    pred_bridges = clf.fit_predict(X) == -1  # 计算哪些点是异常点（桥接点）

    labels = compute_cluster_labels(X, k, pred_bridges, local_indices,
                                    local_distances)

    # 确保聚类类别数满足 n_clusters
    if n_clusters != None:
        unique_labels = np.unique(labels)
        if len(unique_labels) > n_clusters:
            print(f"当前聚类数 {len(unique_labels)} 超过目标 {n_clusters}，进行合并...")
            labels = merge_clusters(labels, n_clusters)

    return labels, pred_bridges


def merge_clusters(labels, n_clusters):
    """ 如果聚类类别数超过 n_clusters，则合并 """
    unique_labels = np.unique(labels)
    if len(unique_labels) <= n_clusters:
        return labels

    # 使用 KMeans 进行聚类合并
    from sklearn.cluster import KMeans
    new_labels = KMeans(n_clusters=n_clusters,
                        random_state=42).fit_predict(labels.reshape(-1, 1))

    return new_labels
