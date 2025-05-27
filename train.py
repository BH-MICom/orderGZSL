import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path
import json
import pickle  # 添加在文件开头的import部分

from core.loss import TumorFocalLoss
from torch.utils.data import WeightedRandomSampler

# 将项目根目录添加到 Python 路径
sys.path.append(str(Path(__file__).parent))
from core.config import parse_args
from core.function import *
from models.networks import PretrainedModel, ResNet3D
from dataset.dataset import APTOSDataset, DRDDataset, ADNIDataset, TumorDataset
from utils.utils import create_logger, delete_files_in_folder, plot_list, save_config_as_json, setup_seed

logger = None
record_logger = None


def init_environment(config):
    """Initialize environment settings such as CUDNN and random seeds."""
    save_config_as_json(config, os.path.join(config.result_dir, "config.json"))

    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    torch.backends.cudnn.deterministic = config.cudnn_deterministic
    torch.backends.cudnn.enabled = config.cudnn_enabled

    setup_seed(config.train_seed)
    device = torch.device(f'cuda:{config.device}' if torch.cuda.is_available() else 'cpu')
    return device
    

def initialize_model(config, device):
    """Initialize the model, criterion, optimizer, and learning rate scheduler."""
    if config.dataset == "APTOS" or config.dataset == "DRD":
        model = PretrainedModel(config).to(device)
    else:
        model = ResNet3D(config).to(device)

    if config.train_stage == 1 and config.pre_train:
        state_dict = torch.load(config.pretrain_path)
        # If model was saved with nn.DataParallel, adjust keys accordingly
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        
        # for param in model.parameters():
        #     param.requires_grad = False

        model.fc = nn.Sequential(nn.Linear(32, 256), nn.ReLU(), nn.Linear(256, 3)).to(device)
        # for param in model.fc.parameters():
        #     param.requires_grad = True
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(
        [{'params': model.parameters(), 'initial_lr': config.train_sgd_lr}],
        lr=config.train_sgd_lr,
        momentum=config.train_sgd_momentum,
        weight_decay=config.train_sgd_weight_decay
    )

    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    return model, criterion, optimizer, lr_scheduler

def get_class_weights(dataset):
    """Calculate the class weights for balanced sampling."""
    # Assuming dataset provides labels directly or a way to get them
    labels = [sample[1].item() for sample in dataset]  # Adjust based on dataset structure

    from collections import Counter

    # Count occurrences of each class
    label_counts = Counter(labels)
    
    # Inverse of frequency as class weights
    class_weights = {label: 1.0 / count for label, count in label_counts.items()}

    # Assign weights to each sample based on its class
    sample_weights = torch.tensor([class_weights[label] for label in labels])

    return sample_weights


def get_datasets(config):
    """Load the training and validation datasets."""

    if config.dataset == "APTOS":
        train_dataset = APTOSDataset(config, logger, mode="train")
        valid_dataset = APTOSDataset(config, logger, mode="valid")
    elif config.dataset == "DRD":
        train_dataset = DRDDataset(config, mode="train")
        valid_dataset = DRDDataset(config, mode="valid")
    elif config.dataset == "ADNI":
        train_dataset = ADNIDataset(config, mode="train", fold=config.fold)
        valid_dataset = ADNIDataset(config, mode="test", fold=config.fold)
    elif config.dataset == "ZDFY":
        train_dataset = TumorDataset(config, mode="train", fold=config.fold)
        valid_dataset = TumorDataset(config, mode="test", fold=config.fold)

    # Get sample weights for training dataset
    sample_weights = get_class_weights(train_dataset)

    # Create a sampler for balanced class sampling
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # Allow replacement to ensure balanced sampling
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,  # Use WeightedRandomSampler
        batch_size=config.train_batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.valid_batch_size,
        shuffle=config.valid_shuffle,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return train_loader, valid_loader


def save_checkpoint(model, output_dir, epoch, acc):
    """Save the model checkpoint."""
    if isinstance(acc, list):
        torch.save(model.state_dict(), os.path.join(output_dir, f'h_{acc[3]:.4f}_acc_{acc[0]:.4f}_seen_{acc[1]:.4f}_unseen_{acc[2]:.4f}_{epoch + 1:02d}.pth'))
    else:
        torch.save(model.state_dict(), os.path.join(output_dir, f'checkpoint{epoch + 1:02d}_{acc}.pth'))


def main(config):
    device = init_environment(config)

    model, criterion, optimizer, lr_scheduler = initialize_model(config, device)

    train_loader, valid_loader = get_datasets(config)

    output_dir = config.result_dir
    train_loss_list = []
    valid_loss_list = []
    train_acc_list = []
    valid_acc_list = []
    seen_acc_list = []
    unseen_acc_list = []
    h_list = []

    train_func = train_features if config.train_stage == 0 else train_classifier
    validate_func = validate_features if config.train_stage == 0 else validate_classifier
    
    best_acc = -1
    best_h = -1
    best_seen_acc = -1
    best_unseen_acc = -1
    
    # 添加用于记录统计量的字典
    stats_history = {
        'mus_norm': {f'class_{i}': [] for i in range(config.num_classes)},
        'sigmas_norm': {f'class_{i}': [] for i in range(config.num_classes)},
        'nablas_norm': {f'class_{i}': [] for i in range(config.num_classes)},
        'means_norm': {f'class_{i}': [] for i in range(config.num_classes)},
        'vars_norm': {f'class_{i}': [] for i in range(config.num_classes)},
        'train_acc': [],
        'valid_acc': [],
        'train_loss': [],
        'valid_loss': [],
    }
    
    for epoch in range(config.train_epoch):
        if config.train_stage == 1:
            train_loss, train_acc, stats = train_func(logger, config, model, train_loader, optimizer, criterion, epoch, device)
            # 记录每个类别的统计量
            for i in range(config.num_classes):
                stats_history['mus_norm'][f'class_{i}'].append(stats['mus_norm'][i].item())
                stats_history['sigmas_norm'][f'class_{i}'].append(stats['sigmas_norm'][i].item())
                stats_history['nablas_norm'][f'class_{i}'].append(stats['nablas_norm'][i].item())
                stats_history['means_norm'][f'class_{i}'].append(stats['means_norm'][i].item())
                stats_history['vars_norm'][f'class_{i}'].append(stats['vars_norm'][i].item())
            stats_history['train_acc'].append(train_acc)
            stats_history['train_loss'].append(train_loss)
        else:
            train_loss, train_acc = train_func(logger, config, model, train_loader, optimizer, criterion, epoch, device)
            
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        lr_scheduler.step()

        valid_loss, acc, seen_acc, unseen_acc, h, _ , _ = validate_func(logger, config, model, valid_loader, criterion, epoch, device)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(acc)
        seen_acc_list.append(seen_acc)
        unseen_acc_list.append(unseen_acc)
        h_list.append(h)

        stats_history['valid_acc'].append(acc)
        stats_history['valid_loss'].append(valid_loss)
        if validate_func == validate_features:
            if (epoch + 1) % 50 == 0:
                save_checkpoint(model, output_dir, epoch, acc)
        else:
            if h >= best_h:
                best_h = h
                save_checkpoint(model, output_dir, epoch, [acc, seen_acc, unseen_acc, h])
            if acc >= best_acc:
                best_acc = acc
                save_checkpoint(model, output_dir, epoch, [acc, seen_acc, unseen_acc, h])
            if seen_acc >= best_seen_acc:
                best_seen_acc = seen_acc
                save_checkpoint(model, output_dir, epoch, [acc, seen_acc, unseen_acc, h])
            if unseen_acc >= best_unseen_acc:
                best_unseen_acc = unseen_acc
                save_checkpoint(model, output_dir, epoch, [acc, seen_acc, unseen_acc, h])

        plot_list(train_loss_list, os.path.join(output_dir, "loss_train.png"))
        plot_list(valid_loss_list, os.path.join(output_dir, "loss_valid.png"))
        plot_list(train_acc_list, os.path.join(output_dir, "acc_train.png"))
        plot_list(valid_acc_list, os.path.join(output_dir, "acc_valid.png"))
        plot_list(seen_acc_list, os.path.join(output_dir, "seen_acc_valid.png"))
        plot_list(unseen_acc_list, os.path.join(output_dir, "unseen_acc_valid.png"))
        plot_list(h_list, os.path.join(output_dir, "h_valid.png"))

    # 在训练结束后保存统计数据和绘制统计图
    if config.train_stage == 1:
        # 保存统计数据到PKL文件
        stats_data = {
            'mus_norm': {
                f'class_{i}': stats_history['mus_norm'][f'class_{i}'] 
                for i in range(config.num_classes)
            },
            'sigmas_norm': {
                f'class_{i}': stats_history['sigmas_norm'][f'class_{i}']
                for i in range(config.num_classes)
            },
            'nablas_norm': {
                f'class_{i}': stats_history['nablas_norm'][f'class_{i}']
                for i in range(config.num_classes)
            },
            'means_norm': {
                f'class_{i}': stats_history['means_norm'][f'class_{i}']
                for i in range(config.num_classes)
            },
            'vars_norm': {
                f'class_{i}': stats_history['vars_norm'][f'class_{i}']
                for i in range(config.num_classes)
            },
            'train_acc': stats_history['train_acc'],
            'valid_acc': stats_history['valid_acc'],
            'train_loss': stats_history['train_loss'],
            'valid_loss': stats_history['valid_loss'],
        }
        
        # 保存为PKL文件
        stats_file = os.path.join(output_dir, 'training_stats.pkl')
        with open(stats_file, 'wb') as f:  # 改为'wb'二进制写入模式
            pickle.dump(stats_data, f)
            
        # 绘制单个综合统计图
        plot_list(
            [stats_history['sigmas_norm'][f'class_{i}'] for i in range(config.num_classes)] + [stats_history['train_acc']], 
            os.path.join(output_dir, "training_statistics.png"),
            labels=[f'Class {i} Std Norm' for i in range(config.num_classes)] + ['Accuracy'],
            title='Training Statistics'
        )


if __name__ == '__main__':
    config = parse_args()
    delete_files_in_folder(config.result_dir)

    logger = create_logger(config.log_dir)
    # os.makedirs(config.result_dir, exist_ok=True)
    main(config)

