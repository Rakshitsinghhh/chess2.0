import torch
import torch.nn as nn
import torch.nn.functional as F
from model.cnn_encoder import CNNEncoder
from model.policy_head import PolicyHead
from model.value_head import ValueHead

class ChessModel(nn.Module):
    def __init__(self, input_channels=20, num_moves=4672):  # Changed to 20
        super().__init__()
        self.input_channels = input_channels
        self.num_moves = num_moves
        
        self.encoder = CNNEncoder(in_channels=input_channels, ch=256)
        self.policy_head = PolicyHead(in_channels=256, num_moves=num_moves)
        self.value_head = ValueHead(in_channels=256)
        
    def forward(self, x):
        # Calculate phase from material (works with 20 channels)
        # Assuming channels 0-5: white pieces, 6-11: black pieces
        piece_values = torch.tensor([1, 3, 3, 5, 9, 0], device=x.device)  # P,N,B,R,Q,K
        
        # Calculate material for each side (simplified for 20 channels)
        if x.shape[1] >= 12:
            white_material = 0
            black_material = 0
            for i, val in enumerate(piece_values):
                if i < 6:
                    white_material += x[:, i].sum(dim=[1, 2]) * val
                    black_material += x[:, i+6].sum(dim=[1, 2]) * val
            total_material = white_material + black_material
            phase = (total_material / 78.0).clamp(0, 1).unsqueeze(1)
        else:
            phase = torch.full((x.size(0), 1), 0.5, device=x.device)
        
        features = self.encoder(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features, phase)
        
        return policy_logits, value

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def load_checkpoint(path, device="cpu"):
        checkpoint = torch.load(path, map_location=device)
        
        # Get model parameters from checkpoint
        model_state = checkpoint["model_state"]
        
        # Try to infer input channels from checkpoint
        input_channels = 20  # Default for your data
        num_moves = 4672
        
        model = ChessModel(input_channels=input_channels, num_moves=num_moves).to(device)
        
        # Load with strict=False to handle potential mismatches
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys[:5]}...")
        
        print(f"Loaded checkpoint: shard {checkpoint.get('shard_id', 'unknown')}, "
              f"epoch {checkpoint.get('epoch', 'unknown')}")
        
        return model, checkpoint.get("optimizer_state"), checkpoint.get("epoch"), checkpoint.get("shard_id")
    
    def save_checkpoint(self, path, optimizer, epoch, shard_id, metrics=None):
        checkpoint = {
            "model_state": self.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "shard_id": shard_id,
            "input_channels": self.input_channels,
            "num_moves": self.num_moves,
        }
        if metrics:
            checkpoint["metrics"] = metrics
        torch.save(checkpoint, path)