import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):

    # Specific convolutional block followed by batch normalization and relu for unet

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.main(x)
        out = self.bn(out)
        out = self.activation(out)
        return out
    

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))