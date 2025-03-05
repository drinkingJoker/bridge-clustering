'''
Author       : Leaf(2064944038@qq.com)
Version      : V1.0
Date         : 2025-01-10 14:33:28
Description  : 
'''
import os
import sys

from PIL import Image
from sklearn.preprocessing import StandardScaler

cur_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(cur_dir)
sys.path.append(parent_dir)

from pathlib import Path
from scipy.io import arff
# import arff
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs

import cv2


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
    assert klass in df.columns and not klass is None
    # 将分类数据转换为整数编码
    df[klass] = pd.factorize(df[klass])[0]

    cols = [x for x in df.columns if x != klass]
    # 确保 X 中的所有元素都是数值类型 (float 或 int)
    X = df[cols].apply(pd.to_numeric, errors='coerce').values
    # 处理可能的 NaN 值，例如用均值填充
    X = np.nan_to_num(X, nan=np.nanmean(X))
    # X = df[cols].values
    labels = df[klass].values
    return X, labels


def read_dataset(fullpath):
    if isinstance(fullpath, str):
        fullpath = Path(fullpath)
    if fullpath.suffix == '.csv':
        df = pd.read_csv(fullpath, sep=',')
        X, y = df[['0', '1']], df['label'].to_numpy()
    elif fullpath.suffix == '.arff':
        X, y = read_arff(fullpath)
    elif fullpath.suffix == '.txt':
        df = pd.read_csv(fullpath, delim_whitespace=True, header=None)
        X, y = df[[0, 1]], df[2].to_numpy()
    else:
        raise ValueError(f'invalid path {fullpath}')
    return X, y


def arff_test():
    fileDir = './data/datasets/'
    artificialDir = fileDir + 'artificial/'

    data = artificialDir + '2d-3c-no123.arff'
    Load_arff(data)


def create_feature_vectors(image):
    """
    将图像转换为特征向量矩阵。
    每个像素点转换为一个5维向量：(R,G,B,x,y)
    """
    # 获取图像尺寸
    height, width = image.shape[:2]

    # 创建坐标网格
    y_coords, x_coords = np.mgrid[0:height, 0:width]

    # 重塑图像和坐标数组
    pixels = image.reshape(-1, 3)  # 将图像重塑为n行3列（BGR值）
    x_coords = x_coords.reshape(-1, 1)  # 将坐标重塑为n行1列
    y_coords = y_coords.reshape(-1, 1)

    # 组合特征向量
    features = np.hstack((pixels, x_coords, y_coords))
    print(features.shape)
    # print(x_coords.shape)
    # print(y_coords.shape)

    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, height, width


def read_image(image_path):
    # 读取图像
    # pil_image = Image.open(image_path)
    # print(pil_image)
    # pil_image.show()
    image = cv2.imread(image_path)  # BGR

    # cv2.imshow('image', image)
    # print(image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    features, height, width = create_feature_vectors(image)
    return image, features, height, width


def image_test():
    datas_dir = parent_dir + '/data/Berkeley Segmentation Dataset 500'
    images_path = datas_dir + '/images'
    trains_path = images_path + '/train'

    image_path = trains_path + '/' + os.listdir(trains_path)[0]
    print(image_path)
    read_image(image_path)


if __name__ == '__main__':
    # arff_test()
    image_test()
