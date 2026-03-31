import torch.nn as nn
import torch.nn.functional as F
from model.residual import ResidualBlock

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=20, ch=256, num_res_blocks=10):  # Changed to 20
        super().__init__()
        self.in_channels = in_channels
        self.ch = ch
        
        # Input projection
        self.input_conv = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        self.input_bn = nn.BatchNorm2d(ch)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(ch) for _ in range(num_res_blocks)]
        )
        
        # Optional: Add attention mechanism
        self.attention = SEBlock(ch)
        
    def forward(self, x):
        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.res_blocks(x)
        x = self.attention(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y