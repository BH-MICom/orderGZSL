import math
import os
try:
    import cv2
except ImportError:
    print("Warning: cv2导入失败，尝试导入headless版本")
    import cv2.cv2 as cv2
    # 如果上面的导入也失败，则表明没有任何可用的OpenCV版本
import numpy as np
import torch
import torch.nn as nn
from models.layers import ConvBlock2D
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.models.video import r3d_18, mc3_18
import torchvision.models as models
from models.resnet import resnet10, resnet18
import nibabel as nib
from scipy.ndimage import binary_dilation


class ResNet3D(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.encoder = resnet18(
                    sample_input_C=self.config.model_in_planes,
                    sample_input_D=self.config.input_D,
                    sample_input_H=self.config.input_H,
                    sample_input_W=self.config.input_W,
                    num_classes=self.config.num_classes
                    )
        
        net_dict = self.encoder.state_dict()
        pretrain = torch.load("/home/xxxx/GZSL/results/resnet_18.pth")
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
            
        net_dict.update(pretrain_dict)
        self.encoder.load_state_dict(net_dict)

        self.middle_fc = nn.Linear(512, self.config.model_features_number)

        self.fc = nn.Sequential(nn.Linear(self.config.model_features_number, 256), nn.ReLU(), nn.Linear(256, self.config.num_classes))
  
        self.mus = nn.Parameter(torch.rand((1, self.config.model_features_number)), requires_grad=True)
        self.deltas = nn.Parameter(torch.rand((self.config.num_classes, self.config.model_features_number)), requires_grad=True)
        self.sigmas = nn.Parameter(torch.rand((self.config.num_classes, self.config.model_features_number)), requires_grad=True)

        nn.init.kaiming_uniform_(self.mus, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.deltas, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.sigmas, a=math.sqrt(5))

    def forward(self, x, labels, mix=False, concat=False, cam=False, encoder=False):
        conv_features = self.encoder(x)
            
        x = self.gap(conv_features)
        features = torch.flatten(x, start_dim=1)

        if encoder is True:
            return features
        
        features = self.middle_fc(features)
        mixed_features_tensor = None
        if mix is True and cam is False:
            device = features.device

            idx_1 = (labels == 0).nonzero(as_tuple=True)[0]
            idx_2 = (labels == 2).nonzero(as_tuple=True)[0]

            num1 = len(idx_1)
            num2 = len(idx_2)
            pairs_to_mix = (num1 + num2) // 2
            new_features = []
            new_labels = []
            if num1 >= 1 and num2 >= 1:
                for _ in range(pairs_to_mix):
                    random_idx_1 = idx_1[torch.randint(num1, (1,))]
                    random_idx_2 = idx_2[torch.randint(num2, (1,))]

                    features_1 = features[random_idx_1]
                    features_2 = features[random_idx_2]

                    mixed_features = features_1 + (features_2 - features_1) * 0.5

                    new_features.append(mixed_features)
                    new_labels.append(torch.tensor([1], device=device))

            if len(new_features): 
                mixed_features_tensor = torch.stack(new_features).view(-1, self.config.model_features_number)
                if concat is True:
                    features = torch.cat([features] + new_features, dim=0)
                    new_labels_tensor = torch.cat(new_labels, dim=0)
                    labels = torch.cat([labels, new_labels_tensor], dim=0)
                    shuffled_indices = torch.randperm(features.size(0))
                    features = features[shuffled_indices] 
                    labels = labels[shuffled_indices]
        
        out = self.fc(features)
        if cam:
            conv_features.requires_grad_(True)  # 确保卷积特征启用梯度追踪
            # 通过out的类别来进行反向传播
            loss = out[torch.arange(out.size(0)), torch.argmax(out, dim=1)].sum()
            assert conv_features.requires_grad
            loss.requires_grad_(True)  # 确保损失启用梯度
            loss.backward()  # 计算梯度
            grads = conv_features.grad  # 获取卷积特征的梯度
            cams = self.compute_gradcam(conv_features, grads)
            return features, out, labels, mixed_features_tensor, cams
        return features, out, labels, mixed_features_tensor
    
    def compute_gradcam(self, conv_features, grads):
        # 计算梯度的全局平均池化
        pooled_grads = torch.mean(grads, dim=[2, 3, 4], keepdim=True)
        # 计算Grad-CAM的权重
        gradcam = torch.relu(torch.sum(pooled_grads * conv_features, dim=1))
        return gradcam
    
    def save_as_nii(self, image, filename):
        import nibabel as nib
        # 假设image是一个numpy数组，形状为 (D, H, W)
        affine = np.eye(4)  # 创建一个单位矩阵作为仿射矩阵
        nii_image = nib.Nifti1Image(image, affine)
        nib.save(nii_image, filename)

    
    def visualize_cam_3d(self, input_image, cam, pred_label, label, idx, name):
        
        nib_data = nib.load(os.path.join("/mnt/xxxx/datasets/ADNIt/register2x", name))
        affine = nib_data.affine

        image = nib_data.get_fdata()
        min_val = np.min(image)
        max_val = np.max(image)

        normalized_image = (image - min_val) / (max_val - min_val)

        # 对image归一化


        # 将CAM调整到输入图像大小
        cam_resized = F.interpolate(
                                    cam.unsqueeze(0).unsqueeze(0), 
                                    size=(image.shape[0], 
                                          image.shape[1], 
                                          image.shape[2]
                                          ), 
                                    mode='trilinear', 
                                    align_corners=False
                                    ).squeeze(0).squeeze(0).cpu().detach().numpy()
        
        # 归一化到 0-1
        if np.max(cam_resized) - np.min(cam_resized) != 0:
            cam_resized = (cam_resized - np.min(cam_resized)) / (np.max(cam_resized) - np.min(cam_resized))
        else:
            cam_resized = np.zeros_like(cam_resized)


        mask = (image*255 > 0.1).astype(np.uint8)

        # 定义一个结构元素（这里使用 3x3x3 立方体结构元素，也可以用其他形状）
        structure = np.ones((5, 5, 5), dtype=np.uint8)

        # 执行膨胀操作
        dilated_mask = binary_dilation(mask, structure=structure)

        cam_resized = cam_resized * dilated_mask

        voxelsize = (1.0, 1.0, 1.0)

        # 将 overlay 转换为 NIfTI 图像
        overlay_nii = nib.Nifti1Image(cam_resized, affine=affine)

        # 修改体素大小 (这是通过 header 来设置的)
        overlay_nii.header.set_zooms(voxelsize)

        # 保存为 NIfTI 文件 (.nii)
        output_path = f'/home/xxxx/adni_cam/{name}_{label}_dilated.nii'
        nib.save(overlay_nii, output_path)

        
        # # 获取批量中指定样本的特定类别的CAM切片
        # cam_specific = cam_resized[0][0]

        # for i in range(1, 2):
            
        #     slice_idx = 34

        #     if i == 1:
        #         image_slice = input_image[0, slice_idx, :, :].cpu().numpy()
        #     elif i == 2:
        #         image_slice = input_image[0, :, slice_idx, :].cpu().numpy()
        #     elif i == 3:
        #         image_slice = input_image[0, :, :, slice_idx].cpu().numpy()

        #     # 保存原始图像
        #     image_path = os.path.join(self.config.result_dir, "cam", f"dim{i}", str(label.item()), f'cam_pred_{str(pred_label.item())}_{idx}_original.png')
        #     os.makedirs(os.path.dirname(image_path), exist_ok=True)
        #     cv2.imwrite(image_path, (image_slice * 255).astype(np.uint8))  # 归一化到 [0, 255] 并保存

        #     # 获取 CAM 图像
        #     cam_specific_slice = cam_specific[slice_idx, :, :].detach().cpu().numpy() if i == 1 else \
        #         cam_specific[:, slice_idx, :].detach().cpu().numpy() if i == 2 else \
        #         cam_specific[:, :, slice_idx].detach().cpu().numpy()
            
        #     # 对cam_specific_slice归一化
        #     cam_specific_slice = (cam_specific_slice - np.min(cam_specific_slice)) / (np.max(cam_specific_slice) - np.min(cam_specific_slice))


        #     # resized_cam_specific_slice = cv2.resize(cam_specific_slice, (182, 218), interpolation=cv2.INTER_CUBIC)

        #     # 将 CAM 图像叠加到原始图像上
        #     cam_image = cv2.applyColorMap((cam_specific_slice * 255).astype(np.uint8), cv2.COLORMAP_JET)  # 使用 JET 调色板

        #     # resized_cam_image = cv2.resize(cam_image, (182, 218), interpolation=cv2.INTER_CUBIC)

        #     original_image = (image_slice * 255).astype(np.uint8)  # 归一化原图像

        #     if len(original_image.shape) == 2:  # 如果是灰度图像
        #         original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)


        #     original_image_path = os.path.join(self.config.result_dir, "heatmap34", f"dim{i}", str(label.item()), f'cam_pred_{str(pred_label.item())}_{idx}_original.png')
        #     os.makedirs(os.path.dirname(original_image_path), exist_ok=True)
        #     cv2.imwrite(original_image_path, original_image)
            

        #     # 创建 brain_mask
        #     threshold = 0.1
        #     brain_mask = (image_slice > threshold).astype(np.float32)

        #     for alpha in [0.2, 0.3, 0.4, 0.5]:
        #         cam_image_resized = cv2.resize(cam_image, (original_image.shape[1], original_image.shape[0]))
        #         overlay_image = cv2.addWeighted(original_image, 1, cam_image_resized, alpha, 0)

        #         cam_image_path = os.path.join(self.config.result_dir, "heatmap34", f"dim{i}", str(label.item()), f'cam_pred_{str(pred_label.item())}_{idx}_{alpha}.png')
        #         os.makedirs(os.path.dirname(cam_image_path), exist_ok=True)
        #         cv2.imwrite(cam_image_path, overlay_image)
        #         self.post_process(cam_image_path, brain_mask)


    def post_process(self, image_path, brain_mask):
        # 读取原始图像
        image = cv2.imread(image_path)
        
        # 将 mask 转换为二值图像
        brain_mask = (brain_mask > 0).astype(np.uint8)

         # 如果图像是彩色图像，确保 mask 适应到每个通道
        if len(image.shape) == 3:  # 彩色图像 (H, W, C)
            masked_image = image * brain_mask[:, :, None]  # 对每个通道应用掩模
        else:  # 灰度图像 (H, W)
            masked_image = image * brain_mask  # 对灰度图像直接应用掩模
        
        # 保存应用 mask 后的图像
        cv2.imwrite(image_path, masked_image)

    def save_visualization(self, input_image, label, cam, out, idx, name):
        # 对输出进行argmax操作，获取最有可能的类别
        pred_label = torch.argmax(out)
        
        self.visualize_cam_3d(input_image, cam, pred_label, label, idx, name)


class PretrainedEncoder(nn.Module):
    def __init__(self, model_name):
        super(PretrainedEncoder, self).__init__()
        self.model_name = model_name

        if self.model_name == "resnet50":   
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        elif self.model_name == "inception_v3":
            inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
            self.encoder = nn.Sequential(
                inception.Conv2d_1a_3x3,
                inception.Conv2d_2a_3x3,
                inception.Conv2d_2b_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2),
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2),
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
            )
        elif self.model_name == "vit":
            vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            self.encoder = nn.Sequential(*list(vit.children())[:-2])
        elif self.model_name == "swin_transformer":
            swin = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)
            self.encoder = nn.Sequential(*list(swin.children())[:-2])
        elif self.model_name == "swin_v2_b":
            swin = models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT)
            self.encoder = nn.Sequential(*list(swin.children())[:-2])

    def forward(self, x):
        x = self.encoder(x)
        return x

class PretrainedModel(nn.Module):
    def __init__(self, config):
        super(PretrainedModel, self).__init__()

        self.config = config
        model_name = config.pretrain_encode_name
        # 使用预训练的编码器
        self.encoder = PretrainedEncoder(model_name)
        # GAP层
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层
        if model_name == "resnet50":
            self.fc = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 5)) 
        elif model_name == "inception_v3":
            self.fc = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 5))
        elif model_name == "vit":
            self.fc = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 5))
        elif model_name == "swin_transformer":
            self.fc = nn.Sequential(nn.Linear(32, 256), nn.ReLU(), nn.Linear(256, 5))
        elif model_name == "swin_v2_b":
            self.fc = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 5))

        self.middle_fc = nn.Linear(1024, 32)


        self.mus = nn.Parameter(torch.rand((1, self.config.model_features_number)), requires_grad=True)
        self.deltas = nn.Parameter(torch.rand((self.config.num_classes, self.config.model_features_number)), requires_grad=True)
        self.sigmas = nn.Parameter(torch.rand((self.config.num_classes, self.config.model_features_number)), requires_grad=True)

        nn.init.kaiming_uniform_(self.mus, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.deltas, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.sigmas, a=math.sqrt(5))

    def forward(self, x, labels, mix=False, concat=False, cam=False):
        x = self.encoder(x)
        x = self.gap(x)
        features = torch.flatten(x, 1)
        features = self.middle_fc(features)

        mixed_features_tensor = None
        if mix is True and cam is False:
            device = features.device

            idx_0 = (labels == 0).nonzero(as_tuple=True)[0]
            idx_2 = (labels == 2).nonzero(as_tuple=True)[0]

            num0 = len(idx_0)
            num2 = len(idx_2)
            pairs_to_mix = (num0 + num2) // 2
            new_features = []
            new_labels = []
            if num0 >= 1 and num2 >= 1:
                for _ in range(pairs_to_mix):
                    random_idx_0 = idx_0[torch.randint(num0, (1,))]
                    random_idx_2 = idx_2[torch.randint(num2, (1,))]

                    features_0 = features[random_idx_0]
                    features_2 = features[random_idx_2]

                    alpha = 0.5

                    # 使用 alpha 进行加权混合
                    mixed_features = alpha * features_0 + (1 - alpha) * features_2

                    new_features.append(mixed_features)
                    new_labels.append(torch.tensor([1], device=device))

            if len(new_features): 
                mixed_features_tensor = torch.stack(new_features).view(-1, self.config.model_features_number)
                if concat is True:
                    features = torch.cat([features] + new_features, dim=0)
                    new_labels_tensor = torch.cat(new_labels, dim=0)
                    labels = torch.cat([labels, new_labels_tensor], dim=0)
                    shuffled_indices = torch.randperm(features.size(0))
                    features = features[shuffled_indices] 
                    labels = labels[shuffled_indices]
        
        out = self.fc(features)
        return features, out, labels, mixed_features_tensor