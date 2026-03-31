import torch.nn as nn
import torch.nn.functional as F

class PolicyHead(nn.Module):
    def __init__(self, in_channels=256, num_moves=4672):
        super().__init__()
        self.num_moves = num_moves
        
        # Single conv layer (simpler, matches your old checkpoint)
        self.conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(32)
        
        self.fc = nn.Linear(32 * 8 * 8, num_moves)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)