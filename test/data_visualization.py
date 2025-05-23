#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author       : Leaf(2064944038@qq.com)
Version      : V1.0
Date         : 2025-01-11 17:27:07
Description  : 多个数据集可视化展示（3x3网格布局）
'''
import os
import sys
import matplotlib
# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 获取当前文件所在的目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上级目录的绝对路径
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

import pandas as pd
import numpy as np
from matplotlib import gridspec, pyplot as plt
import math

from utils.data_utils import read_arff, read_dataset


def plot_dataset(ax, X, labels, title):
    """
    在指定的子图上绘制数据集的聚类结果
    
    参数:
    ax: matplotlib子图对象
    X: 数据集特征
    labels: 聚类标签
    title: 子图标题
    """
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    # ax.set_title(title)
    # 确保子图显示完整的边框
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    # 移除坐标轴刻度，使图像更整洁
    ax.set_xticks([])
    ax.set_yticks([])


def plot_multiple_datasets(dataset_files, save_path=None):
    """
    在一张3x3网格的图上展示多个数据集
    
    参数:
    dataset_files: 数据集文件路径列表
    save_path: 结果保存路径，如果为None则不保存
    """
    # 限制最多显示9个数据集
    if len(dataset_files) > 9:
        dataset_files = dataset_files[:9]
    
    # 计算网格行数和列数
    n_datasets = len(dataset_files)
    n_cols = 3
    n_rows = math.ceil(n_datasets / n_cols)
    
    # 创建图形和网格布局
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(n_rows, n_cols)
    
    # 遍历数据集文件
    for i, dataPath in enumerate(dataset_files):
        # 获取文件名（不含路径）
        filename = os.path.basename(dataPath)
        print(f'处理数据集: {filename}')
        
        # 读取数据集
        X, true_labels = read_dataset(dataPath)
        
        # 创建子图
        row = i // n_cols
        col = i % n_cols
        ax = plt.subplot(gs[row, col])
        
        # 绘制数据集
        plot_dataset(ax, X, true_labels, filename)
    
    # 如果数据集数量少于9，填充空白子图
    for i in range(n_datasets, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = plt.subplot(gs[row, col])
        ax.axis('off')  # 关闭坐标轴
    
    # 设置整体标题
    plt.suptitle('多数据集可视化展示', fontsize=16)
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存结果
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        print(f'结果已保存至: {save_path}')
    
    return fig





if __name__ == '__main__':
    # 数据集路径设置
    data_folder = './data/datasets/'
    artificial_folder = 'synthetic/'
    real_world_folder = 'real-world/'
    select_folder = artificial_folder  # 选择数据集类型
    folder = data_folder + select_folder
    
    # 结果保存路径设置
    save_path = './result/'
    multi_datasets_savepath = save_path + 'multi_datasets/' + select_folder
    # 修改保存配置，增加右侧边距以确保右边框不被裁剪
    plot_config = {'bbox_inches': 'tight', 'pad_inches': 0.1}
    
    # 确保保存目录存在
    if not os.path.exists(multi_datasets_savepath):
        os.makedirs(multi_datasets_savepath)
    
    # 获取所有数据集文件路径
    dataset_files = [folder + filename for filename in os.listdir(folder)]
    
    # 按每9个数据集为一组进行处理
    for i in range(0, len(dataset_files), 9):
        batch_files = dataset_files[i:i+9]
        batch_num = i // 9 + 1
        
        # 绘制数据集可视化结果
        fig = plot_multiple_datasets(batch_files)
        # 保存结果
        save_file = f'{multi_datasets_savepath}batch_{batch_num}_visualization.png'
        fig.savefig(save_file, **plot_config)
        plt.close(fig)
    
    print('所有数据集处理完成！')