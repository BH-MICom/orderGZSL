# -*- coding: utf-8 -*-
import os
import h5py
import csv

import pandas as pd
from torch.functional import norm
from sklearn.model_selection import StratifiedKFold

from utils import *
import numpy as np
import nibabel as nib
import random
import copy
import torch
import torch.nn.functional as F
import torchio as tio
from scipy.ndimage import zoom
from tqdm import tqdm
import sys

modalities = ['FLAIR', 'T1', 'T1C', 'T2']
blacklist = ['TCI1311', 'TCI1734', 'TCI1449', 'TCI0849', '.DS_Store']  # 该数据大小异常


def str2int(num):
    return int(num.replace(' ', ''))


def process(data_dir: str, prior_shape: tuple, out_dir: str):
    names = os.listdir(data_dir)
    images_list = []
    label_list = []

    for name in tqdm(names, desc="running", file=sys.stdout):
        if name in blacklist:
            continue

        # 读取 basic_info，提取 label 信息：三分类/IDH/1p19q/六分类
        with open(os.path.join(data_dir, name, 'basic_info.csv'), 'r') as f:
            info = list(csv.reader(f))[1]
            # 类别编号从 0 开始
            tri_category, six_category, IDH, onep19q = str2int(info[9]) - 1, str2int(info[10]) - 1, str2int(info[13]), str2int(info[14])
            labels = np.array([tri_category, six_category, IDH, onep19q])
            f.close()

        if six_category not in [2, 4, 1]:
            continue

        # 提取 mask
        mask = nib.load(os.path.join(data_dir, name, 'MASK_FLAIR_MNI.nii.gz')).get_fdata()
        # 提取所有图像，缩放到 (96, 96, 64)，并 cat 到 dim0，存储到 name 中
        images = []
        for modality in modalities:
            image_path = os.path.join(data_dir, name, modality + '_MNI.nii.gz')
            image = nib.load(image_path).get_fdata()
            if len(image.shape) > 3:
                logger.warning('警告: ' + '图像尺寸不合法 > ' + str(image.shape))
                image = np.mean(image, axis=(3, 4))
            # tumor_size 归一化到占整脑的比例
            if modality == 'FLAIR':
                tumor_size = np.sum(mask) / np.sum(image != 0)
            if image.shape != mask.shape:
                logger.warning('警告: ' + name + ' 的MASK与图像未对齐 ' + str(image.shape) + ' <> ' + str(mask.shape) + ' 对MASK进行修正')
                mask = crop_pad_to_shape(mask, image.shape)
            image = image * mask  # 剔除非肿瘤区域
            bbox = get_bbox(mask)
            # 提出肿瘤区域
            image = image[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1], bbox[0][2]:bbox[1][2]]
            # 扩充到 指定比例 (非严格)
            image = pad_to_ratio(image, (3, 3, 2))
            # 放缩到指定尺寸
            zoom_ratio = tuple((np.array(prior_shape) / np.array(image.shape)).tolist())
            #image = zoom(image, zoom_ratio)
            image = F.interpolate(torch.from_numpy(image[np.newaxis, np.newaxis]), scale_factor=zoom_ratio, mode='trilinear').numpy()[0][0]
            if image.shape != prior_shape:
                image = crop_pad_to_shape(image, prior_shape)
            image = normalize(image)  # 每个模态的肿瘤单独归一化
            images.append(image)
        images = np.array(images)  # 图像尺寸 (4, 96, 96, 64)
        images_list.append(images)
        label_list.append(labels)

    # 组合成字典
    entries = [{'dataset_name': name, 'data': images, 'label': labels} for name, images, labels in zip(names, images_list, label_list)]
    # 保存为npy文件
    np.save(os.path.join(out_dir, 'entries_selected.npy'), entries)

def generate(data_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # 读取npy文件
    entries = np.load(data_dir, allow_pickle=True)

    images_list = []
    label_list = []
    for entry in entries:
        images_list.append(entry['data'])
        label_list.append(entry['label'][0])


    labels_list = np.array(label_list)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_index = 1
    for train_index, test_index in skf.split(images_list, labels_list):
        train_images = [images_list[i] for i in train_index]
        test_images = [images_list[i] for i in test_index]
        train_labels = [labels_list[i] for i in train_index]
        test_labels = [labels_list[i] for i in test_index]

        # 统计label分布
        train_label_distribution = {0: 0, 1: 0, 2: 0} 
        for label in train_labels:
            train_label_distribution[label] += 1 
        print(f"train label distribution in fold {fold_index}: {train_label_distribution}")

        test_label_distribution = {0: 0, 1: 0, 2: 0}
        for label in test_labels:
            test_label_distribution[label] += 1
        print(f"test label distribution in fold {fold_index}: {test_label_distribution}")

        # 保存数据
        train_store_path = os.path.join(out_dir, f"train_data_fold{fold_index}.h5")
        test_store_path = os.path.join(out_dir, f"test_data_fold{fold_index}.h5")

        save_data(train_images, train_labels, train_store_path)
        save_data(test_images, test_labels, test_store_path)

        fold_index += 1

def save_data(data, label, path):
    with h5py.File(path, 'w') as data_store:
        data_store.create_dataset('data', data=data)
        data_store.create_dataset('label', data=label)
    
if __name__ == '__main__':
    data_dir = "/mnt/xxxx/datasets/ZDFY/match"
    out_dir = "/mnt/xxxx/datasets/ZDFY/5fold_selected_mni"
    os.makedirs(out_dir, exist_ok=True)

    process(data_dir, (96, 96, 64), out_dir)
    generate(os.path.join(out_dir, 'entries_selected.npy'), out_dir)
