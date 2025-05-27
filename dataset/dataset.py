from collections import Counter
import os
import pickle
from random import randint, random, sample
import pandas as pd
import h5py
import torch
import torchio as tio
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
from tqdm import tqdm


class APTOSDataset(Dataset):
    def __init__(self, config, logger, mode='train', transform=None):

        self.config = config
        self.root_path = config.dataset_root
        self.mode = mode

        assert mode in ['train', 'valid', 'test'], "Mode must be 'train', 'valid' or 'test'"

        self.data_file = os.path.join(self.config.dataset_root, f"{mode}.pkl")

        # 读取数据
        with open(self.data_file, 'rb') as f:
            data = pickle.load(f)
            self.images = data['images'].clone().detach().float() / 255.0
            self.labels = data['labels'].clone().detach().long()
            logger.info(f"Images shape: {self.images.shape}")
            # label分布
            logger.info(f"Label distribution: {Counter(self.labels.tolist())}")

        if self.mode == 'train':
            for item in set(self.config.dataset_ordinal) - set(self.config.seen_classes_ordinal):
                self.images = self.images[self.labels != item]
                self.labels = self.labels[self.labels != item]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label
 

class DRDDataset(Dataset):
    def __init__(self, config, mode='train'):
        assert mode in ['train', 'valid', 'test'], "Mode must be 'train', 'valid' or 'test'"
        self.config = config
        self.mode = mode
        self.data_file = os.path.join(self.config.dataset_root, f"{mode}_dataset224_augmented2000.pkl")
        
        # 读取数据
        with open(self.data_file, 'rb') as f:
            data = pickle.load(f)
            self.images = torch.tensor(data['images'], dtype=torch.float32).permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
            self.labels = torch.tensor(data['labels'], dtype=torch.long)
            print("Images shape:", data['images'].shape)
            print("Labels shape:", data['labels'].shape)
        
        if self.mode == 'train':
            for item in set(self.config.dataset_ordinal) - set(self.config.seen_classes_ordinal):
                self.images = self.images[self.labels != item]
                self.labels = self.labels[self.labels != item]
            

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

class TumorDataset(Dataset):
    '''
    mode : train / valid / test
    '''

    def __init__(self, config, mode, fold):
        super().__init__()

        self.config = config
        self.dataset_dir = config.dataset_root
        self.mode = mode
        self.fold = fold

        self.transform = tio.Compose([
            tio.RandomAffine(
                scales=(0.9, 1.2),
                degrees=60,
                isotropic=True,
                p=0.5
            )
        ]) if mode == 'train' else None

        self.open_hdf5()

        self.images = torch.from_numpy(self.images[:]).float()
        self.labels = torch.tensor(self.labels[:], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels['label'].tolist()

    def open_hdf5(self):
        if self.mode == "train":
            self.dataset = h5py.File(os.path.join(self.dataset_dir, f'train_data_fold{self.fold}.h5'), 'r')
            self.images = self.dataset['data']
            self.labels = list(self.dataset['label'])

            self.new_labels = [1 if label == 0 else (0 if label == 1 else 2) for label in self.labels]
            self.labels = self.new_labels

            if self.mode == 'train':
                for item in set(self.config.dataset_ordinal) - set(self.config.seen_classes_ordinal):
                    self.images = np.array([image for image, label in zip(self.images, self.labels) if label != item])
                    self.labels = np.array([label for label in self.labels if label != item])
                    

        if self.mode == "test":
            self.dataset = h5py.File(os.path.join(self.dataset_dir, f'test_data_fold{self.fold}.h5'), 'r')
            self.images = self.dataset['data']
            self.labels = self.dataset['label']

            self.new_labels = [1 if label == 0 else (0 if label == 1 else 2) for label in self.labels]
            self.labels = self.new_labels



    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.mode == 'train':
            image = self.transform(image)

        return image, label

class ADNIDataset(Dataset):
    '''
    mode : train / valid / test
    '''

    def __init__(self, config, mode, fold):
        super().__init__()
        assert mode in ['train', 'test'], "Mode must be 'train' or 'test'"

        self.config = config
        self.dataset_dir = config.dataset_root
        self.mode = mode
        self.fold = fold

        self.transform = tio.Compose([
            tio.RandomAffine(
                scales=(0.9, 1.2),
                degrees=60,
                isotropic=True,
                p=0.5
            )
        ]) if mode == 'train' else None

        self.open_hdf5()

    def __len__(self):
        if self.mode == "train":
            return len(self.keys)
        else:
            return len(self.keys) + len(self.unseen_keys)

    def get_labels(self):
        return self.labels['label'].tolist()

    def open_hdf5(self):
        if self.mode == "train":
            self.original_images = h5py.File(os.path.join(self.dataset_dir, f'train_data_fold{self.fold}.h5'), 'r')
            self.original_labels = pd.read_csv(os.path.join(self.dataset_dir, f'train_labels_fold{self.fold}.csv'))
            self.labels = self.original_labels[self.original_labels['label'] != 1].reset_index(drop=True)
            self.keys = self.labels['dataset_name'].tolist()
            self.images = {key: self.original_images[key] for key in self.keys}

        if self.mode == "test":
            self.images = h5py.File(os.path.join(self.dataset_dir, f'test_data_fold{self.fold}.h5'), 'r')
            self.labels = pd.read_csv(os.path.join(self.dataset_dir, f'test_labels_fold{self.fold}.csv'))
            
            self.keys = self.labels['dataset_name'].tolist()

            self.train_images = h5py.File(os.path.join(self.dataset_dir, f'train_data_fold{self.fold}.h5'), 'r')
            self.train_labels = pd.read_csv(os.path.join(self.dataset_dir, f'train_labels_fold{self.fold}.csv'))
            unseen_labels = self.train_labels[self.train_labels['label'] == 1].reset_index(drop=True)
            self.unseen_keys = unseen_labels['dataset_name'].tolist()

    def __getitem__(self, index):
        
        if self.mode == "train":
            dataset_name = self.keys[index]
            image = self.images[dataset_name]
            label = self.labels[self.labels['dataset_name'] == dataset_name]['label'].values[0]
        else:
            if index < len(self.keys):
                dataset_name = self.keys[index]
                image = self.images[dataset_name]
                label = self.labels[self.labels['dataset_name'] == dataset_name]['label'].values[0]
            else:
                dataset_name = self.unseen_keys[index - len(self.keys)]
                image = self.train_images[dataset_name]
                label = 1

        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)  # Add channel dimension for TorchIO

        image = torch.from_numpy(image).float()

        label = torch.tensor(label, dtype=torch.long)

        if self.mode == "test":
            if index < len(self.keys):
                return image, label, self.keys[index]
            else:
                return image, label, self.unseen_keys[index - len(self.keys)]

        return image, label, self.keys[index]