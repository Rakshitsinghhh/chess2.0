import torch
import torch.nn as nn
from model.cnn_encoder import CNNEncoder
from model.policy_head import PolicyHead
from model.value_head  import ValueHead

class ChessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder     = CNNEncoder()
        self.policy_head = PolicyHead()
        self.value_head  = ValueHead()

    def forward(self, x):
        material = x[:, :12].sum(dim=[1, 2, 3])
        phase    = (material / 32.0).clamp(0, 1).unsqueeze(1)
        features      = self.encoder(x)
        policy_logits = self.policy_head(features)
        value         = self.value_head(features, phase)
        return policy_logits, value

    @staticmethod
    def load_checkpoint(path, device="cpu"):
        checkpoint = torch.load(path, map_location=device)
        model = ChessModel().to(device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded checkpoint: shard {checkpoint['shard_id']}, epoch {checkpoint['epoch']}")
        return model, checkpoint.get("optimizer_state"), checkpoint.get("epoch"), checkpoint.get("shard_id")

    def save_checkpoint(self, path, optimizer, epoch, shard_id):
        torch.save({
            "model_state":     self.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch":           epoch,
            "shard_id":        shard_id,
        }, path)
