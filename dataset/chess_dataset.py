import torch
from torch.utils.data import Dataset
import os

class ChessDataset(Dataset):
    def __init__(self, shards_dir):
        self.data = []

        # Load all shard files
        for file in os.listdir(shards_dir):
            if file.endswith(".pt"):
                shard_path = os.path.join(shards_dir, file)
                shard = torch.load(shard_path)
                self.data.extend(shard)

        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        return {
            "board": sample["board"],   # (12,8,8)
            "move": sample["move"],     # int
            "value": sample["value"],   # float
            "mask": sample["mask"]      # (4672,)
        }