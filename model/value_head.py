import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueHead(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.bn   = nn.BatchNorm2d(1)
        self.fc1  = nn.Linear(64 + 1, 256)
        self.fc2  = nn.Linear(256, 128)
        self.fc3  = nn.Linear(128, 1)

    def forward(self, x, phase=None):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        if phase is not None:
            x = torch.cat([x, phase], dim=1)
        else:
            import torch as t
            dummy = t.full((x.size(0), 1), 0.5, dtype=x.dtype, device=x.device)
            x = torch.cat([x, dummy], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))
