'''
Author       : Leaf(2064944038@qq.com)
Version      : V1.0
Date         : 2025-01-26 14:55:25
Description  : 
'''
import os
import sys

from PIL import Image
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans

cur_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(cur_dir)
sys.path.append(parent_dir)

from utils.data_utils import read_image
from bridge_clustering import BridgeClustering

if __name__ == '__main__':
    datas_dir = parent_dir + '/data/Berkeley Segmentation Dataset 500'
    images_path = datas_dir + '/images'
    trains_path = images_path + '/train'
    select_dir = trains_path

    kls = LocalOutlierFactor
    args = {'contamination': .2, 'n_neighbors': 15}
    k = 7
    for image in os.listdir(select_dir):
        image_path = select_dir + '/' + image
        # image_path = trains_path + '/' + os.listdir(select_dir)[0]
        image, features, height, width = read_image(image_path)

        od = kls(**args)
        clf = BridgeClustering(od, k)
        clf.fit(features)

        # km = KMeans(n_clusters=3)
        # predicted_labels = km.fit_predict(features)

        predicted_labels = clf.labels_
        label = predicted_labels.reshape([width, height])

        #创建一张新的灰度图保存聚类后的结果
        pic_new = Image.new('L', (width, height))
        #根据所属类别向图片中添加灰度值
        # 最终利用聚类中心点的RGB值替换原图中每一个像素点的值，便得到了最终的分割后的图片
        for i in range(width):
            for j in range(height):
                pic_new.putpixel((i, j), int(256 / (label[i][j] + 1)))

        #以JPEG格式保存图片
        # pic_new.save("result_demo1.jpg","JPEG")
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(image)
        axes[1].imshow(pic_new)
        plt.show()
