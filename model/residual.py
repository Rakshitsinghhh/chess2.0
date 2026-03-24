import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels=256, se_ratio=4):
        super().__init__()
        self.conv1  = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1    = nn.BatchNorm2d(channels)
        self.conv2  = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2    = nn.BatchNorm2d(channels)
        self.se_fc1 = nn.Linear(channels, channels // se_ratio)
        self.se_fc2 = nn.Linear(channels // se_ratio, channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        se  = out.mean(dim=[2, 3])
        se  = F.relu(self.se_fc1(se))
        se  = torch.sigmoid(self.se_fc2(se))
        out = out * se.unsqueeze(-1).unsqueeze(-1)
        return F.relu(out + residual)
