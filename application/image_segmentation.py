'''
Author       : Leaf(2064944038@qq.com)
Version      : V1.0
Date         : 2025-01-26 14:55:25
Description  : 
'''
import os
import sys

from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans

cur_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(cur_dir)
sys.path.append(parent_dir)

from utils.function import merge_clusters
from utils.data_utils import read_image
from bridge_clustering import BridgeClustering

if __name__ == '__main__':
    datas_dir = parent_dir + '/data/Berkeley Segmentation Dataset 500'
    images_path = datas_dir + '/images'
    trains_path = images_path + '/train'
    select_dir = trains_path

    save_path = './result/'
    clusters_savepath = save_path + 'image_segmentation/'
    plot_config = {'bbox_inches': 'tight', 'pad_inches': 0}

    kls = LocalOutlierFactor
    args = {'contamination': .2, 'n_neighbors': 15}
    k = 7
    for image in os.listdir(select_dir):
        image_path = select_dir + '/' + image
        # image_path = trains_path + '/' + os.listdir(select_dir)[0]
        # img, data, height, width = read_image(image_path)

        # 读取图像
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 图像转换为一维数据
        data = img.reshape((-1, 3)).astype(np.float32)

        od = kls(**args)
        clf = BridgeClustering(od, k)
        labels = clf.fit_predict(data)

        # clf2 = BridgeClustering(od, k, 2)
        # clf4 = BridgeClustering(od, k, 4)
        # clf8 = BridgeClustering(od, k, 8)
        # clf16 = BridgeClustering(od, k, 16)
        # clf32 = BridgeClustering(od, k, 32)
        # clf64 = BridgeClustering(od, k, 64)
        # clf128 = BridgeClustering(od, k, 128)

        # labels = clf2.fit_predict(data)
        labels2 = merge_clusters(labels, 2)
        # 生成图像
        centers2 = np.array(
            [np.mean(data[labels2 == i], axis=0) for i in range(2)])
        centers2 = np.uint8(centers2)
        res2 = centers2[labels2]
        segmented_img2 = res2.reshape(img.shape)

        # labels = clf4.fit_predict(data)
        labels4 = merge_clusters(labels, 4)
        centers4 = np.array(
            [np.mean(data[labels4 == i], axis=0) for i in range(4)])
        centers4 = np.uint8(centers4)
        res4 = centers4[labels4]
        segmented_img4 = res4.reshape(img.shape)

        # labels = clf8.fit_predict(data)
        labels8 = merge_clusters(labels, 8)
        centers8 = np.array(
            [np.mean(data[labels8 == i], axis=0) for i in range(8)])
        centers8 = np.uint8(centers8)
        res8 = centers8[labels8]
        segmented_img8 = res8.reshape(img.shape)

        # labels = clf16.fit_predict(data)
        labels16 = merge_clusters(labels, 16)
        centers16 = np.array(
            [np.mean(data[labels16 == i], axis=0) for i in range(16)])
        centers16 = np.uint8(centers16)
        res16 = centers16[labels16]
        segmented_img16 = res16.reshape(img.shape)

        # labels = clf32.fit_predict(data)
        labels32 = merge_clusters(labels, 32)
        centers32 = np.array(
            [np.mean(data[labels32 == i], axis=0) for i in range(32)])
        centers32 = np.uint8(centers32)
        res32 = centers32[labels32]
        segmented_img32 = res32.reshape(img.shape)

        # labels = clf64.fit_predict(data)
        labels64 = merge_clusters(labels, 64)
        centers64 = np.array(
            [np.mean(data[labels64 == i], axis=0) for i in range(64)])
        centers64 = np.uint8(centers64)
        res64 = centers64[labels64]
        segmented_img64 = res64.reshape(img.shape)

        # labels = clf128.fit_predict(data)
        labels128 = merge_clusters(labels, 128)
        centers128 = np.array(
            [np.mean(data[labels128 == i], axis=0) for i in range(128)])
        centers128 = np.uint8(centers128)
        res128 = centers128[labels128]
        segmented_img128 = res128.reshape(img.shape)

        #图像转换为RGB显示
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dst2 = cv2.cvtColor(segmented_img2, cv2.COLOR_BGR2RGB)
        dst4 = cv2.cvtColor(segmented_img4, cv2.COLOR_BGR2RGB)
        dst8 = cv2.cvtColor(segmented_img8, cv2.COLOR_BGR2RGB)
        dst16 = cv2.cvtColor(segmented_img16, cv2.COLOR_BGR2RGB)
        dst32 = cv2.cvtColor(segmented_img32, cv2.COLOR_BGR2RGB)
        dst64 = cv2.cvtColor(segmented_img64, cv2.COLOR_BGR2RGB)
        dst128 = cv2.cvtColor(segmented_img128, cv2.COLOR_BGR2RGB)

        #用来正常显示中文标签
        plt.rcParams['font.sans-serif'] = ['SimHei']
        #显示图像
        titles = [
            u'原始图像', u'聚类图像 K=2', u'聚类图像 K=4', u'聚类图像 K=8', u'聚类图像 K=16',
            u'聚类图像 K=32', u'聚类图像 K=64', u'聚类图像 K=128'
        ]
        images = [img, dst2, dst4, dst8, dst16, dst32, dst64, dst128]
        fig = plt.figure()
        for i in range(8):
            plt.subplot(2, 4, i + 1),
            plt.imshow(images[i], 'gray'),
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()
        # plt.closekk(fig)
        # cluster_savepath = f'{clusters_savepath}{image}.png'
        # fig.savefig(cluster_savepath, **plot_config)
