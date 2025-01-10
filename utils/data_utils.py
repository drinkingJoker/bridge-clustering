'''
Author       : Leaf(2064944038@qq.com)
Version      : V1.0
Date         : 2025-01-10 14:33:28
Description  : 
'''
from scipy.io import arff
# import arff
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs


def generate_data():
    # 生成一些示例数据
    X, true_labels = make_blobs(n_samples=300,
                                centers=4,
                                cluster_std=0.6,
                                random_state=0)
    return X, true_labels


def Load_arff(file):
    # 使用 liac-arff 解析 ARFF 文件，并指定编码为 UTF-8
    with open(file, 'r', encoding='utf-8') as file:
        decoded_content = arff.load(file)

    # 获取属性名称和数据
    attributes = decoded_content['attributes']
    data = decoded_content['data']

    # print(data)
    # print(meta)
    # print(data.shape)

    # 分离数据为 X 和 true_labels
    # points = np.array([(item[0], item[1]) for item in data])
    # labels = np.array([item[2] for item in data])
    num_features = len(attributes) - 1
    points = np.array([row[:num_features] for row in data])
    labels = np.array([row[num_features] for row in data])

    if isinstance(labels[0], bytes):
        label_mapping = {
            label.decode('utf-8'): idx
            for idx, label in enumerate(
                set(label.decode('utf-8') for label in labels))
        }
        labels = np.array(
            [label_mapping[label.decode('utf-8')] for label in labels],
            dtype=int)
    elif isinstance(labels[0], str):
        label_mapping = {label: idx for idx, label in enumerate(set(labels))}
        labels = np.array([label_mapping[label] for label in labels],
                          dtype=int)
    return points, labels


def read_arff(file):
    data, meta = arff.loadarff(file)
    df = pd.DataFrame(data)

    klass = None
    if 'class' in df.columns:
        klass = 'class'
    elif 'CLASS' in df.columns:
        klass = 'CLASS'
    elif 'Class' in df.columns:
        klass = 'Class'
    # 将分类数据转换为整数编码
    df[klass] = pd.factorize(df[klass])[0]

    cols = [x for x in df.columns if x != klass]
    X = df[cols].values
    y = df[klass].values
    return X, y


if __name__ == '__main__':
    fileDir = './data/datasets/'
    artificialDir = fileDir + 'artificial/'

    data = artificialDir + '2d-3c-no123.arff'
    Load_arff(data)
