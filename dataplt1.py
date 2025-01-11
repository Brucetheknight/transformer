import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def load_and_preprocess_data(file_path, mjd_min, mjd_max, window_size=10, forecast_steps=1):
    """
    加载并预处理数据，包括数据清洗、筛选、归一化及生成滑动窗口数据集
    Args:
        file_path (str): 数据文件路径
        mjd_min (int): 数据筛选的最小MJD
        mjd_max (int): 数据筛选的最大MJD
        window_size (int): 滑动窗口大小
        forecast_steps (int): 预测步数

    Returns:
        tuple: 返回训练集、测试集数据及scaler
    """
    # 数据导入与清理
    data = pd.read_excel(file_path, header=1)
    data['MJD(days)'] = pd.to_numeric(data['MJD(days)'], errors='coerce')  # 转换为数值型
    data['PT-TT（s）'] = pd.to_numeric(data['PT-TT（s）'], errors='coerce')
    data = data.dropna(subset=['MJD(days)', 'PT-TT（s）'])  # 去除NaN行

    # 筛选数据范围
    filtered_data = data[(data['MJD(days)'] >= mjd_min) & (data['MJD(days)'] <= mjd_max)]

    # 提取时间序列数据
    y = filtered_data['PT-TT（s）'].values.reshape(-1, 1)
    time_info = filtered_data['MJD(days)'].values.reshape(-1, 1)  # 时间戳信息

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler.fit_transform(y)  # 归一化目标数据

    # 滑动窗口生成数据（包括时间戳作为额外特征）
    def create_sliding_window(data, time_info, window_size, forecast_steps=1):
        X, Y = [], []
        for i in range(len(data) - window_size - forecast_steps + 1):
            X.append(data[i:i + window_size])  # 滑动窗口数据
            time_window = time_info[i:i + window_size]  # 对应的时间戳
            X[-1] = np.concatenate((X[-1], time_window), axis=-1)  # 将时间戳信息作为一个额外特征
            Y.append(data[i + window_size:i + window_size + forecast_steps])  # 目标值
        return np.array(X), np.array(Y)

    X, Y = create_sliding_window(y_scaled, time_info, window_size, forecast_steps)

    # 按时间顺序划分训练集和测试集
    train_size = int(len(X) * 1)
    X_real, Y_real = X[:train_size], Y[:train_size]

    return X_real, Y_real, scaler