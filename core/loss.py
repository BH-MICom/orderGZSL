import torch
import torch.nn as nn

class TumorFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(TumorFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, output, target):
        ce_loss = self.ce(output, target)
        pt = torch.exp(-ce_loss)
        if self.alpha is None:
            focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        else:
            focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
