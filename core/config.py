import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Model Configuration')

    # Base directories
    parser.add_argument('--base_dir', type=str, default='/home/xxxx/GZSL', help='Base directory')
    parser.add_argument('--result_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--sub_result_dir', type=str, default='/home/xxxx/GZSL/results/ADNI_xxx/stage1_3', help='Sub result directory')
    parser.add_argument('--log_dir', type=str, default='log', help='Log directory')

    # flexible settings
    parser.add_argument('--train_stage', type=int, default=1, help='0 means train features, 1 means train classifier')
    parser.add_argument('--fold', type=int, default=4, help='Fold number')
    parser.add_argument('--pre_train', type=bool, default=True, help='Wether to pretrain')
    parser.add_argument('--pretrain_path', type=str, default="/home/xxxx24/GZSL/results/ADNI_xxx/checkpoint200_0.pth", help='the model path of stage 0')
    parser.add_argument('--pretrain_encoder', type=bool, default=True, help='Whether to use the encoder of pretrain model')
    parser.add_argument('--pretrain_encode_name', type=str, default='swin_transformer', help='the name of pretrain encoder')
    parser.add_argument('--ordinal_method', type=str, default='kl', help='the method of ordinal')

    # General settings
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--device', type=int, default=0, help='Device ID')
    parser.add_argument('--print_freq', type=int, default=10, help='Print frequency')
 
    # CUDNN settings
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='CUDNN benchmark')
    parser.add_argument('--cudnn_deterministic', type=bool, default=False, help='CUDNN deterministic')
    parser.add_argument('--cudnn_enabled', type=bool, default=True, help='CUDNN enabled')

    # Model settings
    parser.add_argument('--model_in_planes', type=int, default=1, help='Model input planes')
    parser.add_argument('--input_D', type=int, default=182, help='Input depth')
    parser.add_argument('--input_H', type=int, default=218, help='Input height')
    parser.add_argument('--input_W', type=int, default=182, help='Input width')
    parser.add_argument('--model_encoder', type=int, nargs='+', default=[4, 8, 16, 32], help='Model encoder layers')
    parser.add_argument('--model_features_number', type=int, default=32, help='Model features number')

    # Dataset settings
    parser.add_argument('--dataset', type=str, default='ADNI', help='Dataset')
    parser.add_argument('--dataset_root', type=str, default='/mnt/xxxx/datasets/ADNI/generated_91_109_91', help='Dataset root')
    parser.add_argument('--dataset_ordinal', type=int, nargs='+', default=[0, 1, 2], help='Dataset ordinal order')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--seen_classes_ordinal', type=int, nargs='+', default=[0, 2], help='Seen classes in ordinal')

    # Training settings
    parser.add_argument('--train_seed', type=int, default=12345, help='Training seed')
    parser.add_argument('--train_sgd_lr', type=float, default=1e-3, help='SGD learning rate')
    parser.add_argument('--train_sgd_momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--train_sgd_weight_decay', type=float, default=1e-3, help='SGD weight decay')
    parser.add_argument('--train_batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--train_shuffle', type=bool, default=True, help='Shuffle training data')
    parser.add_argument('--train_epoch', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr_dec_ep', type=int, default=1, help='LR decay for every 100 epochs')

    # Validation settings
    parser.add_argument('--valid_batch_size', type=int, default=256, help='Validation batch size')
    parser.add_argument('--valid_shuffle', type=bool, default=True, help='Shuffle validation data')

    args = parser.parse_args()

    # Assign directories
    args.result_dir = os.path.join(args.base_dir, args.result_dir, args.sub_result_dir)
    args.log_dir = os.path.join(args.base_dir, 'log')

    return args

