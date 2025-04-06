#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ARFF文件加载器
用于解析和加载ARFF格式的数据集文件
"""

import re
import numpy as np
import pandas as pd


def load_arff_file(file_path):
    """
    加载ARFF格式的数据集文件
    
    参数:
        file_path: ARFF文件路径
        
    返回:
        pandas DataFrame对象
    """
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析ARFF文件结构
    relation_match = re.search(r'@RELATION\s+(.+)', content, re.IGNORECASE)
    relation_name = relation_match.group(1).strip() if relation_match else 'unknown'
    
    # 提取属性定义
    attribute_pattern = r'@ATTRIBUTE\s+([^\s]+)\s+([^\s]+)(\s+\{(.+)\})?'
    attribute_matches = re.findall(attribute_pattern, content, re.IGNORECASE)
    
    # 构建列名和数据类型
    columns = []
    dtypes = {}
    categorical_cols = []
    
    for match in attribute_matches:
        attr_name = match[0].strip()
        attr_type = match[1].strip().upper()
        columns.append(attr_name)
        
        # 处理分类属性
        if match[3]:
            categories = [c.strip() for c in match[3].split(',')]
            categorical_cols.append((attr_name, categories))
        elif attr_type in ['NUMERIC', 'REAL', 'INTEGER']:
            dtypes[attr_name] = np.float64
        else:
            # 默认作为字符串处理
            dtypes[attr_name] = str
    
    # 提取数据部分
    data_match = re.search(r'@DATA\s+([\s\S]+)', content, re.IGNORECASE)
    if not data_match:
        raise ValueError("ARFF文件中未找到@DATA部分")
    
    data_content = data_match.group(1).strip()
    
    # 将数据转换为CSV格式
    lines = [line.strip() for line in data_content.split('\n') if line.strip() and not line.strip().startswith('%')]
    
    # 创建DataFrame
    df = pd.DataFrame([line.split(',') for line in lines], columns=columns)
    
    # 转换数据类型
    for col, dtype in dtypes.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except:
                print(f"警告：无法将列 {col} 转换为 {dtype} 类型")
    
    # 处理分类变量
    for col, categories in categorical_cols:
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=categories)
    
    return df