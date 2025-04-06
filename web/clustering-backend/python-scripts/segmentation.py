#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像分割脚本
用于对图像进行分割并生成可视化结果

用法：
    python segmentation.py <input_file> <output_file> <algorithm> [--param value]

参数：
    input_file: 输入图像文件路径
    output_file: 输出图像文件路径
    algorithm: 分割算法（clustering, watershed, grabcut）
    --param value: 算法特定参数
"""

import sys
import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
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


def load_image(file_path):
    """
    加载图像文件
    """
    try:
        # 读取图像
        image = cv2.imread(file_path)
        if image is None:
            raise Exception(f"无法读取图像文件: {file_path}")

        # 转换为RGB颜色空间（OpenCV默认是BGR）
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image
    except Exception as e:
        print(f"加载图像失败: {str(e)}")
        sys.exit(1)


def generate_distinct_colors(n_clusters):
    """
    生成鲜艳的颜色列表，用于聚类结果可视化
    """
    # 使用HSV颜色空间生成均匀分布的颜色
    colors = []
    for i in range(n_clusters):
        # 在HSV空间中，H值决定颜色，均匀分布可以得到不同的颜色
        h = i / n_clusters
        # 饱和度和亮度设置为较高的值，使颜色鲜艳
        s = 0.8
        v = 0.9
        # 转换为RGB
        r, g, b = plt.cm.hsv(h)[0:3]
        colors.append([int(r*255), int(g*255), int(b*255)])
    return np.array(colors, dtype=np.uint8)

def kmeans_segmentation(image, params):
    """
    基于kmeans聚类的图像分割
    """
    # 获取参数
    k = int(params.get('k', 5))
    color_space = params.get('colorSpace', 'rgb').lower()

    # 转换颜色空间
    if color_space == 'hsv':
        image_processed = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif color_space == 'lab':
        image_processed = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    else:  # 默认使用RGB
        image_processed = image.copy()

    # 重塑图像为二维数组，每行是一个像素
    height, width, channels = image_processed.shape
    pixels = image_processed.reshape(-1, channels)

    # 应用K-Means聚类
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)

    # 生成鲜艳的颜色
    distinct_colors = generate_distinct_colors(k)
    
    # 创建分割结果（使用鲜艳的颜色而不是原图像的颜色）
    segmented = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(k):
        segmented[labels.reshape(height, width) == i] = distinct_colors[i]

    # 创建带有标签的图像（用于可视化）
    labeled_image = labels.reshape(height, width)

    return segmented, labeled_image


def watershed_segmentation(image, params):
    """
    分水岭分割
    """
    # 获取参数
    distance_threshold = int(params.get('distanceThreshold', 10))
    apply_smoothing = params.get('applySmoothing', 'true').lower() == 'true'

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 应用平滑处理（可选）
    if apply_smoothing:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 应用二值化
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 噪声去除
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 确定背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 计算距离变换
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # 确定前景区域
    _, sure_fg = cv2.threshold(
        dist_transform, distance_threshold * 0.01 * dist_transform.max(), 255,
        0)
    sure_fg = sure_fg.astype(np.uint8)

    # 寻找未知区域
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 标记
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 应用分水岭算法
    markers = cv2.watershed(image, markers)
    
    # 获取唯一的标记值（不包括-1，它是边界）
    unique_markers = np.unique(markers)
    unique_markers = unique_markers[unique_markers >= 0]  # 排除边界标记-1
    n_segments = len(unique_markers)
    
    # 生成鲜艳的颜色
    distinct_colors = generate_distinct_colors(n_segments)
    
    # 创建分割结果（使用鲜艳的颜色）
    segmented = np.zeros_like(image)
    for i, marker_value in enumerate(unique_markers):
        segmented[markers == marker_value] = distinct_colors[i % len(distinct_colors)]
    
    # 边界标记为红色
    segmented[markers == -1] = [255, 0, 0]

    return segmented, markers


def grabcut_segmentation(image, params):
    """
    GrabCut分割
    """
    # 获取参数
    iter_count = int(params.get('iterCount', 5))
    mode = params.get('mode', 'rect')

    # 创建掩码、背景和前景模型
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # 设置矩形区域（默认使用图像中心区域）
    height, width = image.shape[:2]
    rect = (width // 4, height // 4, width // 2, height // 2)

    # 应用GrabCut算法
    if mode == 'rect':
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iter_count,
                    cv2.GC_INIT_WITH_RECT)
    else:  # 掩码模式（简化处理，实际应用中可能需要用户交互）
        # 创建一个简单的掩码，将中心区域标记为可能的前景
        mask[height // 4:3 * height // 4,
             width // 4:3 * width // 4] = cv2.GC_PR_FGD
        cv2.grabCut(image, mask, None, bgd_model, fgd_model, iter_count,
                    cv2.GC_INIT_WITH_MASK)

    # 创建分割结果
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # 使用鲜艳的颜色来显示前景和背景
    # 生成两种颜色：一种用于前景，一种用于背景
    distinct_colors = generate_distinct_colors(2)
    
    # 创建分割结果（使用鲜艳的颜色）
    segmented = np.zeros_like(image)
    segmented[mask2 == 1] = distinct_colors[0]  # 前景
    segmented[mask2 == 0] = distinct_colors[1]  # 背景

    return segmented, mask


def bridge_segmentation(image, params):
    """
    基于桥点聚类的图像分割
    使用已实现的BridgeClustering类
    """
    # 获取参数
    n_clusters = int(params.get(
        'n_clusters', 8)) if params.get('n_clusters') is not None else 8
    contamination = float(
        params.get('contamination',
                   0.2)) if params.get('contamination') is not None else 0.2
    n_neighbors_od = int(
        params.get('n_neighbors_od',
                   15)) if params.get('n_neighbors_od') is not None else 15
    n_neighbors = int(params.get(
        'n_neighbors', 7)) if params.get('n_neighbors') is not None else 7
    color_space = params.get('colorSpace', 'rgb').lower()

    # 转换颜色空间
    if color_space == 'hsv':
        image_processed = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif color_space == 'lab':
        image_processed = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    else:  # 默认使用RGB
        image_processed = image.copy()

    print("bridge_segmentatio:")

    # 图像转换为一维数据
    data = image_processed.reshape((-1, 3)).astype(np.float32)

    # 选择异常检测方法
    od_method = params.get('outlierDetection', 'LOF')
    if od_method == 'LOF':
        od = LocalOutlierFactor(n_neighbors=n_neighbors_od,
                                contamination=contamination)
    elif od_method == 'IF':
        od = IsolationForest(contamination=contamination, random_state=42)
    else:
        # 默认使用LOF
        od = LocalOutlierFactor(n_neighbors=n_neighbors_od,
                                contamination=contamination)

    # 创建BridgeClustering实例并执行聚类
    bridge_clustering = BridgeClustering(
        od, n_neighbors, None if n_clusters == 1 else n_clusters)
    labels = bridge_clustering.fit_predict(data)

    # 生成鲜艳的颜色
    distinct_colors = generate_distinct_colors(n_clusters)
    
    # 创建分割结果（使用鲜艳的颜色而不是原图像的颜色）
    height, width, _ = image.shape
    segmented = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(n_clusters):
        segmented[labels.reshape(height, width) == i] = distinct_colors[i]

    # 创建带有标签的图像（用于可视化）
    height, width, _ = image.shape
    labeled_image = labels.reshape(height, width)

    return segmented, labeled_image


def visualize_segmentation(original, segmented, labeled_image, output_file):
    """
    可视化分割结果并保存为图像
    """
    plt.figure(figsize=(12, 6))

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title('原始图像')
    plt.axis('off')

    # 显示分割结果
    plt.subplot(1, 2, 2)
    plt.imshow(segmented)
    plt.title('分割结果')
    plt.axis('off')

    # 保存图像
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"分割结果已保存至: {output_file}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='图像分割')
    parser.add_argument('input_file', help='输入图像文件路径')
    parser.add_argument('output_file', help='输出图像文件路径')
    parser.add_argument('algorithm',
                        help='分割算法 (kmeans, watershed, grabcut, bridge)')

    # 添加kmeans聚类分割参数
    parser.add_argument('--k', type=int, help='聚类数量 (kmeans)')
    parser.add_argument('--colorSpace', help='颜色空间 (kmeans, bridge)')

    # 添加分水岭分割参数
    parser.add_argument('--distanceThreshold',
                        type=int,
                        help='标记距离 (watershed)')
    parser.add_argument('--applySmoothing', help='平滑处理 (watershed)')

    # 添加GrabCut分割参数
    parser.add_argument('--iterCount', type=int, help='迭代次数 (grabcut)')
    parser.add_argument('--mode', help='模式 (grabcut)')

    # 添加桥点聚类参数
    parser.add_argument('--n_clusters', type=int, help='聚类数量 (bridge)')
    parser.add_argument('--contamination', type=float, help='异常点比例 (bridge)')
    parser.add_argument('--n_neighbors_od',
                        type=int,
                        help='局部异常因子的邻居数量 (bridge)')
    parser.add_argument('--n_neighbors', type=int, help='桥点聚类的邻居数量 (bridge)')
    parser.add_argument('--outlierDetection', help='异常检测方法 (bridge): LOF或IF')
    parser.add_argument('--useSpatial', help='是否使用空间信息 (bridge): true或false')

    # 解析参数
    args = parser.parse_args()

    # 将参数转换为字典
    params = vars(args)

    # 加载图像
    print(f"加载图像: {args.input_file}")
    image = load_image(args.input_file)
    print(f"图像大小: {image.shape}")

    # 执行分割
    print(f"执行{args.algorithm}分割...")
    if args.algorithm == 'kmeans':
        segmented, labeled_image = kmeans_segmentation(image, params)
    elif args.algorithm == 'watershed':
        segmented, labeled_image = watershed_segmentation(image, params)
    elif args.algorithm == 'grabcut':
        segmented, labeled_image = grabcut_segmentation(image, params)
    elif args.algorithm == 'bridge':
        segmented, labeled_image = bridge_segmentation(image, params)
    else:
        print(f"错误：不支持的分割算法 {args.algorithm}")
        sys.exit(1)

    # 可视化结果
    print("生成可视化结果...")
    visualize_segmentation(image, segmented, labeled_image, args.output_file)


if __name__ == "__main__":
    main()
