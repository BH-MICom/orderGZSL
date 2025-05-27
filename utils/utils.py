import argparse
from collections import Counter
import json
import os
import shutil
from venv import logger
from matplotlib import pyplot as plt
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import time


def label2onehot(label_value, num_classes):
    """
    将 label 值转换为 one-hot 向量
    """
    if type(label_value) != torch.Tensor:
        label_value = torch.Tensor([label_value]).to(torch.long)
    # scatter_(dim, index, src, reduce=None) → Tensor
    # Writes all values from the tensor src into self at
    # the indices specified in the index tensor. 
    one_hot = torch.zeros(num_classes).scatter_(0, label_value, 1).to(torch.long)

    return one_hot


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_bbox(mask: np.ndarray) -> tuple:
    '''
    计算所有肿瘤区域的bbox
    '''
    loc = np.where(mask == 1)
    y_min, y_max = np.min(loc[0]), np.max(loc[0])
    x_min, x_max = np.min(loc[1]), np.max(loc[1])
    z_min, z_max = np.min(loc[2]), np.max(loc[2])

    return ((y_min, x_min, z_min), (y_max, x_max, z_max))


def pad_to_shape(image: np.ndarray, shape: tuple) -> np.ndarray:
    '''
    填充图像到指定形状
    '''
    ori_shape = image.shape
    shape = np.array(shape)
    before = (shape - ori_shape) // 2
    after = (shape - ori_shape) - before
    image = np.pad(image, ((before[0], after[0]), (before[1], after[1]), (before[2], after[2])))

    return image


def pad_to_ratio(image: np.ndarray, ratio: tuple):
    '''
    填充到指定比例
    image : 肿瘤的 bounding box，每个轴向的每个切片都包含肿瘤区域，因此只能作填充，不能做裁剪
    ratio : (y_ratio, x_ratio, z_ratio), default : (3, 3, 2)
    '''
    assert ratio == (3, 3, 2), '只写了 (3, 3, 2) 比例的填充'
    ori_shape = image.shape
    # y, x轴向先填充到相同尺寸，然后判断z轴向
    # 若 z 轴向小于目标大小，直接填充，否则，再填充y,x轴向到目标大小
    yx_len = max(ori_shape[0], ori_shape[1])
    image = pad_to_shape(image, (yx_len, yx_len, ori_shape[2]))
    if ori_shape[2] * 1.5 < yx_len:
        image = pad_to_shape(image, (yx_len, yx_len, int(yx_len / 1.5)))
    else:
        image = pad_to_shape(image, (int(1.5 * ori_shape[2]), int(1.5 * ori_shape[2]), ori_shape[2]))

    return image


def crop_pad_to_shape(image: np.ndarray, shape: tuple):
    '''
    先裁剪mask多出来的部分，再pad缺少的部分
    '''
    # 先裁剪多出来的部分
    ori_shape = np.array(image.shape)
    new_shape = np.array(shape)
    bias = np.clip((ori_shape - new_shape), 0, None)
    image = image[bias[0] // 2:ori_shape[0] - (bias[0] - bias[0] // 2), bias[1] // 2:ori_shape[1] - (bias[1] - bias[1] // 2),
            bias[2] // 2:ori_shape[2] - (bias[2] - bias[2] // 2)]
    # 再填补缺少的部分
    image = pad_to_shape(image, new_shape)

    return image


def get_tumor_position(mask: np.ndarray) -> np.ndarray:
    '''
    计算肿瘤位置：将全脑图像划分为 3x3x3 区域，计算肿瘤属于哪些区域，返回one-hot编码（27位）
    需要 split 三次
    '''
    # 原始 mask 大小不一定是 27 倍数，先截取到 27 倍数
    ori_shape = np.array(mask.shape)
    bias = ori_shape - (ori_shape // 27) * 27
    mask = mask[bias[0] // 2:ori_shape[0] - (bias[0] - bias[0] // 2), bias[1] // 2:ori_shape[1] - (bias[1] - bias[1] // 2),
           bias[2] // 2:ori_shape[2] - (bias[2] - bias[2] // 2)]
    mask = np.split(mask, 3, axis=0)
    mask = [np.split(mask[i], 3, axis=1) for i in range(len(mask))]
    mask = [e for x in mask for e in x]  # unpack 数组 x 是 mask 中元素 (list)，e 是 x 中元素
    mask = [np.split(mask[i], 3, axis=2) for i in range(len(mask))]
    mask = [np.sum(e) != 0 for x in mask for e in x]  # unpack 数组，检查切分的 27 块中每块是否包含肿瘤

    return np.array(mask).astype(int).tolist()


def normalize(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


class AverageMeter(object):
    """
    log changes of target data, you can get current value,
    average value and sum
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, times=1):
        self.val = val
        self.sum += val * times
        self.count += times
        self.avg = self.sum / self.count


def create_logger(out_dir):
    logger = logging.getLogger("base_logger")
    logger.setLevel(logging.INFO)

    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    logfile = os.path.join(out_dir, f'{timestamp}.log')

    os.makedirs(out_dir, exist_ok=True)

    handler = logging.FileHandler(logfile, mode='w')
    handler.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger

def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states['state_dict'], os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, f'model_best_{filename}'))


def delete_files_in_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' was not found and has been created.")
        return  # 文件夹创建后直接返回
    else:
        print("文件夹存在，确定要清空内容吗？（Y/N）")
        yon = input()
        if yon == "N":
            exit()

    # 遍历指定文件夹
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            # 如果是文件，则删除
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
                print(f"Deleted file : {file_path}")
            # 如果是目录，则删除该目录及其所有内容
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"Deleted folder : {file_path}")
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def load_data_as_arrays(dataset):
    images = []
    labels = []
    for i in range(len(dataset)):
        image, _, label = dataset[i]
        images.append(image.numpy())
        labels.append(label.item())
    return np.array(images), np.array(labels)


def plot_list(data_lists, save_path, labels=None, title=None):
    """
    绘制一个或多个数据列表的折线图
    
    Args:
        data_lists: 单个列表或列表的列表，包含要绘制的数据
        save_path: 图片保存路径
        labels: 可选，每条线的标签列表
        title: 可选，图表标题
    """
    plt.figure(figsize=(10, 6))
    
    # 确保data_lists是列表的列表
    if not isinstance(data_lists[0], list):
        data_lists = [data_lists]
    
    # 如果没有提供标签，则使用默认标签
    if labels is None:
        labels = [f'Series {i+1}' for i in range(len(data_lists))]
    
    for data, label in zip(data_lists, labels):
        plt.plot(data, label=label)
    
    if title:
        plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def save_config_as_json(config, file_path):
    config_dict = vars(config) if isinstance(config, argparse.Namespace) else dict(config)

    with open(file_path, 'w') as json_file:
        json.dump(config_dict, json_file, indent=4)

def check_class_distribution(targets, required_classes):
    """
    统计每个类别的样本数量
    """
    class_counts = Counter(targets.cpu().numpy())
    missing_classes = set(required_classes) - set(class_counts.keys())
    if missing_classes:
        logger.warning(f"Missing classes: {missing_classes}")
        return False
    return True