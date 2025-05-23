import os
import sys
# 获取当前文件所在的目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上级目录的绝对路径
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

from bridge_clustering import BridgeClustering
from utils.function import compute_neighbors, determine_bridges
from utils.data_utils import read_arff, read_dataset


def plot(filename, X, bridges, cluster_labels):
    # labels must be a boolean array
    not_bridges = ~bridges

    fig = plt.figure()
    plt.title(filename)
    plt.scatter(
        X[:, 0][not_bridges],
        X[:, 1][not_bridges],
        c=cluster_labels[not_bridges],
        marker='o',
    )
    plt.scatter(X[:, 0][bridges],
                X[:, 1][bridges],
                c='red',
                marker='^',
                alpha=.5)
    plt.axis('off')
    plt.show()
    return fig


if __name__ == '__main__':
    data_folder = './data/datasets/'
    artificial_folder = data_folder + 'synthetic/'
    real_world_folder = data_folder + 'real-world/'
    folder = artificial_folder

    save_path = './result/'
    bridges_savepath = save_path + 'bridges/'
    outliers_savepath = save_path + 'outliers/'
    plot_config = {'bbox_inches': 'tight', 'pad_inches': 0}

    kls = LocalOutlierFactor
    args = {'contamination': .2, 'n_neighbors': 15}
    k = 7

    # for filename in os.listdir(folder):
    for i in range(0, 1):
        filename = 'diamond9.arff'
        dataPath = folder + filename
        print(dataPath)
        X, cluster_labels = read_dataset(dataPath)
        distances, indices = compute_neighbors(X, k)
        bridges = determine_bridges(cluster_labels, indices)

        od = kls(**args)
        clf = BridgeClustering(od, k)
        clf.fit(X)

        predicted_labels = clf.labels_
        outliers = clf.outliers_

        bridge_fig = plot(filename, X, bridges, predicted_labels)
        outlier_fig = plot(filename, X, outliers, predicted_labels)

        bridge_savepath = f'{bridges_savepath}{filename}_bridge_{k}.png'
        outlier_savepath = f'{outliers_savepath}{filename}_outlier_{k}.png'

        bridge_fig.savefig(bridge_savepath, **plot_config)
        outlier_fig.savefig(outlier_savepath, **plot_config)
