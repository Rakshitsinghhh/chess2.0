import torch.nn as nn
import torch.nn.functional as F
from model.residual import ResidualBlock

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=20, channels=256, num_blocks=10):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.input_bn   = nn.BatchNorm2d(channels)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.res_blocks(x)
        return x
