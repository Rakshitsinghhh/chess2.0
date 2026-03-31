import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueHead(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        
        # Single conv layer (matches your old checkpoint)
        self.conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(32)
        
        # Calculate flattened size: 32 channels * 8 * 8 = 2048
        self.flatten_size = 32 * 8 * 8
        
        # FC layers
        self.fc1 = nn.Linear(self.flatten_size + 1, 256)  # +1 for phase
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, phase=None):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)  # [batch, 2048]
        
        if phase is None:
            phase = torch.full((x.size(0), 1), 0.5, device=x.device, dtype=x.dtype)
        
        if phase.dim() == 1:
            phase = phase.unsqueeze(1)
        
        x = torch.cat([x, phase], dim=1)  # [batch, 2049]
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.tanh(self.fc3(x))