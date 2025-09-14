import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.backends.cudnn as cudnn
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.config import parse_args
from models.networks import ResNet3D
from utils.utils import create_logger, delete_files_in_folder, setup_seed
from dataset.dataset import ADNIDataset

def main(logger, portion: str):       
    config = parse_args()
    setup_seed(config.train_seed)

    device = torch.device(f'cuda:{config.device}' if torch.cuda.is_available() else 'cpu')

    model = ResNet3D(config).to(device)
    
    model.load_state_dict(
        torch.load('/home/xxxx/GZSL/results/ADNI/zero-shot/1miss/pres18/fold4/classification/h_0.8723_acc_0.8824_seen_0.8880_unseen_0.8571_67.pth')
    )

    model.requires_grad_(False)

    valid_dataset = ADNIDataset(config, mode="test", fold=4)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.valid_batch_size,
        shuffle=config.valid_shuffle,
        num_workers=config.num_workers,
        pin_memory=True
    )

    model.eval()
    all_features = []
    all_targets = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(valid_loader):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            features, final, targets, mixed_features = model(inputs, targets, mix=True, concat=False)
            all_features.append(features.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_features.append(mixed_features.cpu().numpy())
            all_targets.append(np.array([3]*len(mixed_features)))
            
    all_features = np.concatenate(all_features, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 使用 t-SNE 将特征降维到二维
    tsne = TSNE(n_components=2, perplexity=50)
    features_2d = tsne.fit_transform(all_features)

    filtered_features_2d = features_2d
    filtered_targets = all_targets
    
    # 创建可视化图
    plt.figure(figsize=(9, 6))
    
    # 绘制不同类别的点
    for class_idx, color in zip([0, 1, 2], ['blue', 'orange', 'green']):
        indices = filtered_targets == class_idx
        plt.scatter(filtered_features_2d[indices, 0], filtered_features_2d[indices, 1], label=f'Class {class_idx}', c=color)

    # 将黑色点改为三角形
    black_indices = filtered_targets == 3
    plt.scatter(filtered_features_2d[black_indices, 0], filtered_features_2d[black_indices, 1], 
                label='Class 3', c='black', marker='^')

    # 去除坐标轴等内容
    plt.axis('off')
    
    plt.savefig('/home/xxxx/GZSL/results/tsne_2d2.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == '__main__':
    main(None, 'portion-1')
