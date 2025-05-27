import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

def load_and_plot_statistics(pkl_path, save_dir):
    # 读取PKL文件
    with open(pkl_path, 'rb') as f:
        stats = pickle.load(f)
    
    # 获取epoch数量
    num_epochs = len(stats['sigmas_norm']['class_0'])
    epochs = range(num_epochs)
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 绘制三个类别的标准差范数
    for i in range(3):
        plt.plot(epochs, stats['vars_norm'][f'class_{i}'], 
                label=f'Class {i} Std Norm', 
                linestyle='-')
    
    # 计算并绘制类间距差异
    nablas_diff_01 = np.array(stats['means_norm']['class_1']) - np.array(stats['means_norm']['class_0'])
    nablas_diff_12 = np.array(stats['means_norm']['class_2']) - np.array(stats['means_norm']['class_1'])
    
    plt.plot(epochs, nablas_diff_01, 
            label='Nablas(1-0)', 
            linestyle='--')
    plt.plot(epochs, nablas_diff_12, 
            label='Nablas(2-1)', 
            linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    save_path = os.path.join(save_dir, 'detailed_statistics_post.png')
    plt.savefig(save_path)
    plt.close()
    
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    # 设置PKL文件路径和保存目录
    pkl_path = "/home/chenlb/GZSL/results/rubuttal/test300/training_stats.pkl"  # 替换为实际的PKL文件路径
    save_dir = "/home/chenlb/GZSL/results/rubuttal/test300/"      # 替换为实际的保存目录
    
    load_and_plot_statistics(pkl_path, save_dir) 