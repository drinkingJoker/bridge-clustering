'''
Author       : Leaf(2064944038@qq.com)
Version      : V1.0
Date         : 2025-01-11 17:27:07
Description  : 比较不同聚类算法的性能，生成双Y轴柱状图，横轴为算法名，左侧纵轴为时间，右侧纵轴为准确率(ACC)
'''
import os
import sys
import time

# 直接从acc.py导入cluster_acc函数
from acc import cluster_acc

# 获取当前文件所在的目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上级目录的绝对路径
parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

import matplotlib
# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN, OPTICS
import hdbscan

from bridge_clustering import BridgeClustering
from utils.border_peel.border_peeling import BorderPeel as BorderPeeling
from utils.autoclust import AUTOCLUST
from utils.data_utils import read_dataset

def run_algorithm(algorithm_name, X, cluster_labels, params=None):
    """
    运行指定的聚类算法并返回标签、运行时间和ACC值
    
    参数:
        algorithm_name: 算法名称
        X: 数据集特征矩阵
        cluster_labels: 真实的聚类标签
        params: 算法参数字典
    
    返回:
        labels: 聚类结果标签
        runtime: 运行时间（秒）
        acc: 聚类准确率
    """
    start_time = time.time()
    
    if algorithm_name == 'BAC':
        # 桥点聚类算法
        kls = LocalOutlierFactor
        args = params or {'contamination': .2, 'n_neighbors': 15}
        k = 7
        od = kls(**args)
        clf = BridgeClustering(od, k)
        clf.fit(X)
        labels = clf.labels_
    
    elif algorithm_name == 'DBSCAN':
        # DBSCAN算法
        args = params or {'min_samples': 5}
        dbscan = DBSCAN(**args)
        labels = dbscan.fit_predict(X)
    
    elif algorithm_name == 'OPTICS':
        # OPTICS算法
        args = params or {'min_samples': 0.07, 'cluster_method': 'xi'}
        optics = OPTICS(**args)
        labels = optics.fit_predict(X)
    
    elif algorithm_name == 'HDBSCAN':
        # HDBSCAN算法
        args = params or {'min_cluster_size': 15, 'min_samples': 5}
        hdbscan_clusterer = hdbscan.HDBSCAN(**args)
        labels = hdbscan_clusterer.fit_predict(X)
    
    elif algorithm_name == 'BorderPeeling':
        # BorderPeeling算法
        args = params or {'k': 20}
        border_peeling = BorderPeeling(**args)
        border_peeling.fit(X)
        labels = border_peeling.labels_
    
    elif algorithm_name == 'AUTOCLUST':
        # AUTOCLUST算法
        autoclust = AUTOCLUST()
        autoclust.fit(X)
        labels = autoclust.labels_
    
    else:
        raise ValueError(f"未知算法: {algorithm_name}")
    
    end_time = time.time()
    runtime = end_time - start_time
    
    # 计算ACC
    acc = cluster_acc(cluster_labels, labels)
    
    return labels, runtime, acc


def plot_performance_comparison(algorithm_avg_results):
    """
    绘制算法平均性能比较的双Y轴柱状图
    
    参数:
        algorithm_avg_results: 字典，键为算法名称，值为(平均运行时间, 平均ACC)元组
    
    返回:
        fig: matplotlib图像对象
    """
    algorithms = list(algorithm_avg_results.keys())
    avg_runtimes = [algorithm_avg_results[alg][0] for alg in algorithms]
    avg_accs = [algorithm_avg_results[alg][1] for alg in algorithms]
    
    # 创建图像和主坐标轴
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 设置x轴标签位置
    x = np.arange(len(algorithms))
    width = 0.35
    
    # 绘制平均运行时间柱状图（左Y轴）
    bars1 = ax1.bar(x - width/2, avg_runtimes, width, label='平均运行时间 (秒)', color='skyblue')
    ax1.set_xlabel('聚类算法')
    ax1.set_ylabel('平均运行时间 (秒)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # 创建共享X轴的第二个Y轴
    ax2 = ax1.twinx()
    
    # 绘制平均ACC柱状图（右Y轴）
    bars2 = ax2.bar(x + width/2, avg_accs, width, label='平均ACC (准确率)', color='lightcoral')
    ax2.set_ylabel('平均ACC (准确率)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 设置X轴刻度标签
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms)
    
    # 添加图例
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # 为柱状图添加数值标签
    def add_labels(bars, ax, format_str):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(format_str.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=0)
    
    add_labels(bars1, ax1, '{:.2f}')
    add_labels(bars2, ax2, '{:.4f}')
    
    # 设置标题
    plt.title('聚类算法平均性能比较')
    plt.tight_layout()
    
    return fig


if __name__ == '__main__':
    # 数据集设置
    data_folder = './data/datasets/'
    artificial_folder = 'synthetic/'
    real_world_folder = 'real-world/'
    select_folder = artificial_folder  # 选择数据集类型
    folder = data_folder + select_folder
    
    # 结果保存路径
    save_path = './result/'
    performance_charts_path = save_path + 'performance_charts/' + select_folder
    os.makedirs(performance_charts_path, exist_ok=True)
    
    # 图表保存配置
    plot_config = {'bbox_inches': 'tight', 'pad_inches': 0}
    
    # 要比较的算法列表
    algorithms = ['BAC', 'DBSCAN', 'OPTICS', 'HDBSCAN', 'BorderPeeling', 'AUTOCLUST']
    
    # 创建结果DataFrame
    results_columns = ['dataset'] + [f"{alg}_time" for alg in algorithms] + [f"{alg}_acc" for alg in algorithms]
    results_df = pd.DataFrame(columns=results_columns)
    
    # 用于计算平均性能的字典
    algorithm_all_results = {alg: {'times': [], 'accs': []} for alg in algorithms}
    
    # 处理每个数据集
    for filename in os.listdir(folder):
        dataPath = folder + filename
        print(f"处理数据集: {filename}")
        
        # 读取数据集
        X, cluster_labels = read_dataset(dataPath)
        
        # 存储各算法的结果
        result_row = {'dataset': filename}
        
        # 运行每个算法并记录性能
        for algorithm in algorithms:
            print(f"运行算法: {algorithm}")
            try:
                labels, runtime, acc = run_algorithm(algorithm, X, cluster_labels)
                
                # 保存到结果行
                result_row[f"{algorithm}_time"] = runtime
                result_row[f"{algorithm}_acc"] = acc
                
                # 添加到平均性能计算字典
                algorithm_all_results[algorithm]['times'].append(runtime)
                algorithm_all_results[algorithm]['accs'].append(acc)
                
                print(f"  - 运行时间: {runtime:.2f}秒, ACC: {acc:.4f}")
            except Exception as e:
                print(f"  - 错误: {e}")
                result_row[f"{algorithm}_time"] = np.nan
                result_row[f"{algorithm}_acc"] = np.nan
        
        # 添加结果行到DataFrame
        results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)
    
    # 计算每个算法的平均性能
    algorithm_avg_results = {}
    for algorithm in algorithms:
        times = algorithm_all_results[algorithm]['times']
        accs = algorithm_all_results[algorithm]['accs']
        
        # 计算平均值，忽略NaN值
        avg_time = np.nanmean(times) if times else np.nan
        avg_acc = np.nanmean(accs) if accs else np.nan
        
        algorithm_avg_results[algorithm] = (avg_time, avg_acc)
        print(f"{algorithm} - 平均运行时间: {avg_time:.2f}秒, 平均ACC: {avg_acc:.4f}")
    
    # 绘制平均性能比较图
    fig = plot_performance_comparison(algorithm_avg_results)
    avg_chart_path = f"{save_path}average_performance_comparison.png"
    fig.savefig(avg_chart_path, **plot_config)
    plt.close(fig)
    print(f"平均性能比较图已保存到: {avg_chart_path}")
        
    print("所有数据集处理完成!")