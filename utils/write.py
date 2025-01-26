import os
import pandas as pd


def UpdateARI(df, filename, algorithm_name, ari):
    if df.empty:
        # 如果 DataFrame 是空的，我们先确定要添加的算法名作为初始列
        df[algorithm_name] = None

    # 如果 'filename' 不存在于 DataFrame 的索引中，则添加新行并填充为 NaN
    if filename not in df.index:
        df.loc[filename] = None  # 此操作现在是安全的，因为我们至少有一列存在

    # 如果 'algorithm_name' 不存在于 DataFrame 的列中，则添加新列
    if algorithm_name not in df.columns:
        df[algorithm_name] = None  # 添加新列，值为 NaN

    # 更新对应位置的 ARI 值
    df.at[filename, algorithm_name] = ari


def read_performance_csv(path):
    # 尝试读取现有的 CSV 文件内容到 DataFrame 中
    try:
        df = pd.read_csv(path, index_col=0)
    except FileNotFoundError:
        # 如果文件不存在，则创建一个新的 DataFrame
        df = pd.DataFrame()
    return df


def write_performance_csv(df, path):
    # 确保存储结果的目录存在，如果不存在则创建
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # 将更新后的 DataFrame 写回到 CSV 文件中
    df.to_csv(path)
