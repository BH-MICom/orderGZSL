from models.networks import PretrainedModel, ResNet3D
from core.config import parse_args
from core.function import debug
from dataset.dataset import APTOSDataset, DRDDataset, ADNIDataset, TumorDataset
from utils.utils import create_logger, setup_seed
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import os

config = parse_args()
logger = None


fold = 4

# pretrain_dir = f'/home/xxxx/GZSL/results/ADNI/zero-shot/1miss/pres18/fold{fold}/classification/'
# # 选择这个dir的最后一个文件
# pretrain_path = os.path.join(pretrain_dir, os.listdir(pretrain_dir)[-1])
# print(pretrain_path)
pretrain_path = '/home/xxxx/GZSL/results/ADNI/zero-shot/1miss/no_ordinal/fold4/h_0.6897_acc_0.8889_seen_0.9680_unseen_0.5357_115.pth'


def main(logger, portion: str):
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    torch.backends.cudnn.deterministic = config.cudnn_deterministic
    torch.backends.cudnn.enabled = config.cudnn_enabled

    setup_seed(config.train_seed)

    device = torch.device(f'cuda:{config.device}' if torch.cuda.is_available() else 'cpu')

    if config.dataset == "APTOS" or config.dataset == "DRD":
        model = PretrainedModel(config).to(device)
    else:
        model = ResNet3D(config).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.load_state_dict(
        torch.load(pretrain_path)
    )

    model.requires_grad_(False)
    
    if config.dataset == "APTOS":
        valid_dataset = APTOSDataset(config, logger, mode='test')
    elif config.dataset == "DRD":
        valid_dataset = DRDDataset(config, mode='test')
    elif config.dataset == "ADNI":
        valid_dataset = ADNIDataset(config, mode='test', fold=fold)
    elif config.dataset == "ZDFY":
        valid_dataset = TumorDataset(config, mode='test', fold=fold)
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=len(valid_dataset),
        num_workers=config.num_workers,
        pin_memory=True
    )

    debug(logger, config, model, valid_loader, pretrain_path)


if __name__ == '__main__':
    logger = create_logger(config.log_dir)
    main(logger, 'portion-1')
