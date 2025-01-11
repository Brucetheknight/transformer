import torch
from torch.utils.data import DataLoader, TensorDataset
from transformer import TransformerWithTimeInfo
from dataplt1 import load_and_preprocess_data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error  # R²和RMSE计算

# 超参数设置
BS = 512  # 批量大小
def main() -> None:
    # 数据加载和预处理
    file_path = r"C:\Users\sanluo\Desktop\R及Python\参考数据\2024“ShuWei Cup”_Problem\2024_“ShuWei Cup”C_Problem\Attachment 1.xlsx"
    mjd_min = 56085
    mjd_max = 56185
    window_size = 10
    forecast_steps = 1

    # 加载数据
    X_real, Y_real, scaler = load_and_preprocess_data(
        file_path, mjd_min, mjd_max, window_size, forecast_steps
    )
    # 数据转为PyTorch张量
    forcast_set = TensorDataset(
        torch.tensor(X_real, dtype=torch.float32),
        torch.tensor(Y_real, dtype=torch.float32)
    )
    # 数据加载器（不打乱顺序）
    forecast_loader = DataLoader(forcast_set, batch_size=BS, shuffle=False)  # 保持时间顺序

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载整个模型
    model = torch.load('model_full.pth')  # 直接加载整个模型
    model.eval()  # 切换到评估模式

    # 模型评估（全范围预测）
    model.eval()
    predictions = []
    targets = []
    time_stamps = []  # 用于存储时间戳

    # 将整个数据范围内的预测结果与真实值结合
    with torch.no_grad():
        # 首先，获取训练集的预测（确保时间顺序一致）
        for src, tgt in forecast_loader:
            src, tgt = src.to(device), tgt.to(device)
            time_stamps_batch = src[:, -1, -1].cpu().numpy()  # 获取时间戳部分
            pred = model(src, tgt)
            predictions.append(pred.cpu())
            targets.append(tgt.cpu())
            time_stamps.append(time_stamps_batch)

    # 合并所有预测值、目标值和时间戳
    predictions = torch.cat(predictions, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()
    time_stamps = np.concatenate(time_stamps, axis=0)  # 合并时间戳

    # 转为二维数据
    predictions = predictions.reshape(-1, predictions.shape[-1])
    targets = targets.reshape(-1, targets.shape[-1])

    # 计算R²和RMSE
    r2 = r2_score(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # 可视化预测结果与真实值
    # 保证时间戳范围为[mjd_min, mjd_max]
    plt.figure(figsize=(10, 5))
    plt.plot(time_stamps, targets.flatten(), label='True Values', alpha=0.7)
    plt.plot(time_stamps, predictions.flatten(), label='Predicted Values', alpha=0.7)
    plt.xlabel('MJD (days)')
    plt.ylabel('PT-TT (s)')
    plt.title('True vs Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.xlim([mjd_min, mjd_max])  # 设置x轴范围
    # 在图中添加 R² 和 RMSE
    plt.text(
        mjd_min + (mjd_max - mjd_min) * 0.05,  # 文本 x 位置
        max(targets.flatten()) * 0.9,  # 文本 y 位置
        f"R²: {r2:.4f}\nRMSE: {rmse:.4f}",  # 文本内容
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7)  # 添加背景框
    )
    pdf_path_13 = r"E:\AI\Anaconda3\A\envs\FirstDL\ShuiWeiC\plot_13.pdf"
    plt.savefig(pdf_path_13)
    plt.close()

    plt.figure(figsize=(10, 5))
    # 使用线条连接并加上数据点标记
    plt.plot(time_stamps, targets.flatten(), 'o-', label='True Values', alpha=0.7, markersize=4)
    plt.plot(time_stamps, predictions.flatten(), 'x-', label='Predicted Values', alpha=0.7, markersize=4)
    # 设置图表标签和标题
    plt.xlabel('MJD (days)')
    plt.ylabel('PT-TT (s)')
    plt.title('True vs Predicted Values')
    plt.legend()
    plt.grid(True)
    # 设置x轴范围
    plt.xlim([mjd_min, mjd_max])
    # 在图中添加 R² 和 RMSE
    plt.text(
        mjd_min + (mjd_max - mjd_min) * 0.05,  # 文本 x 位置
        max(targets.flatten()) * 0.9,  # 文本 y 位置
        f"R²: {r2:.4f}\nRMSE: {rmse:.4f}",  # 文本内容
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7)  # 添加背景框
    )
    #显示图表
    pdf_path_14 = r"E:\AI\Anaconda3\A\envs\FirstDL\ShuiWeiC\plot_14.pdf"
    plt.savefig(pdf_path_14)
    plt.close()


if __name__ == "__main__":
    main()