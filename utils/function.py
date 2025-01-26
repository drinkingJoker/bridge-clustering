from collections import defaultdict
import heapq
import numpy as np
from sklearn.neighbors import NearestNeighbors


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


def expand_labels(local_indices, is_bridge_candidate):
    n_points = local_indices.shape[0]
    cluster_labels = np.ones(n_points) * -1
    bridge_indexes = np.arange(n_points)[is_bridge_candidate]
    x_axis = np.arange(n_points)[..., np.newaxis]

    jump_matrix = np.zeros((n_points, n_points), dtype=bool)

    assert ((local_indices == x_axis) == False).all()  # 保证没有自连接

    jump_matrix[x_axis, local_indices] = True
    jump_matrix[x_axis, bridge_indexes] = False

    transposed_jump_matrix = jump_matrix.T
    jump_matrix = jump_matrix | transposed_jump_matrix
    # 确保 jump_matrix 中的每个位置 (i, j) 和 (j, i) 都至少有一个是 True，从而使得 jump_matrix 成为一个对称矩阵。

    jump_matrix[is_bridge_candidate, :] = False
    jump_matrix[:, is_bridge_candidate] = False

    assert (
        np.diag(jump_matrix) == False).all()  # 确保 jump_matrix 的对角线元素全部为 False
    # check_jump_matrix(jump_matrix, local_indices, is_bridge_candidate)

    index_line = np.arange(n_points)

    label = 0
    for i in range(n_points):
        if is_bridge_candidate[i]:
            continue
        if cluster_labels[i] != -1:
            continue
        pq = [i]

        while pq:
            el = pq.pop()
            cluster_labels[el] = label
            elements_to_explore = index_line[jump_matrix[el] &
                                             (cluster_labels == -1)].tolist()
            pq.extend(elements_to_explore)

        label += 1

    return cluster_labels


def generate_jump_distance_matrix(local_distances, local_indices):
    npoints = local_indices.shape[0]
    jump_matrix = np.zeros((npoints, npoints), dtype=float)

    index_array = np.arange(npoints, dtype=int)
    jump_matrix[index_array[..., np.newaxis],
                local_indices[np.newaxis, ...]] = local_distances

    mask = jump_matrix == 0
    transposed_jump_matrix = jump_matrix.T
    jump_matrix = jump_matrix + transposed_jump_matrix * mask

    assert (jump_matrix == jump_matrix.T).all()
    return jump_matrix


def assign_bridge_labels(cluster_labels, local_indices, local_distances):

    def _collect_data(jump_matrix, is_bridge_candidate, index):
        not_ignore = jump_matrix > 0
        is_bridge_candidate = is_bridge_candidate & not_ignore
        not_bridge_candidate = (~is_bridge_candidate) & not_ignore
        neigh_indexes = np.arange(is_bridge_candidate.size, dtype=int)
        index_list = np.ones_like(neigh_indexes) * index

        assert (cluster_labels[neigh_indexes[is_bridge_candidate]] == -1).all()
        assert (cluster_labels[neigh_indexes[not_bridge_candidate]]
                != -1).all()

        assert (jump_matrix[not_bridge_candidate] > 0).all()
        assert (jump_matrix[is_bridge_candidate] > 0).all()

        valid_queue = list(
            zip(jump_matrix[not_bridge_candidate],
                neigh_indexes[not_bridge_candidate], index_list))
        invalid_queue = list(
            zip(jump_matrix[is_bridge_candidate],
                neigh_indexes[is_bridge_candidate], index_list))

        return valid_queue, invalid_queue

    index_array = np.arange(cluster_labels.size)
    is_bridge_candidate = cluster_labels == -1
    missing_indexes = index_array[is_bridge_candidate]
    jump_distance_matrix = generate_jump_distance_matrix(
        local_distances, local_indices)
    # verify_jump_distance_matrix(local_distances, local_indices, jump_distance_matrix)

    pq = []
    pq2 = defaultdict(list)
    for i in missing_indexes:
        jump_row = jump_distance_matrix[i]

        valid_queue, invalid_queue = _collect_data(jump_row,
                                                   is_bridge_candidate, i)
        for t in invalid_queue:
            pq2[t[1]].append(t)

        pq.extend(valid_queue)

    heapq.heapify(pq)

    while pq:
        t = heapq.heappop(pq)
        neigh, me = int(t[1]), int(t[2])
        if cluster_labels[me] != -1:
            continue
        assert cluster_labels[neigh] != -1

        cluster_labels[me] = cluster_labels[neigh]
        to_add = list(filter(lambda x: cluster_labels[x[2]] == -1, pq2[me]))
        heapq.heapify(to_add)

        pq = list(heapq.merge(pq, to_add))
    return cluster_labels


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


def predict_and_cluster(clf, X, k, local_indices=None, local_distances=None):
    # clf.fit(X)
    # pred_bridges = clf.labels_ == 1
    # print(X)
    pred_bridges = clf.fit_predict(X) == -1
    return compute_cluster_labels(X, k, pred_bridges, local_indices,
                                  local_distances), pred_bridges
