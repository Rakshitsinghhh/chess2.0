# import torch
# import chess

# shard = torch.load("data/shards/shard_000.pt", weights_only=False)

# sample = shard[0]
# print(f"Mask shape:      {sample['mask'].shape}")
# print(f"Mask dtype:      {sample['mask'].dtype}")
# print(f"Mask max value:  {sample['mask'].max().item()}")
# print(f"Mask min value:  {sample['mask'].min().item()}")
# print(f"Non-zero count:  {sample['mask'].nonzero().shape[0]}")
# print(f"Raw mask values: {sample['mask'][:20]}")  # first 20 values
# import torch

# shard = torch.load("outputs/models/shard_000.pt", weights_only=False)

# with open("shard_000_summary.txt", "w") as f:
#     f.write(f"Total samples: {len(shard)}\n")
#     for i, sample in enumerate(shard):
#         f.write(f"\n--- Sample {i} ---\n")
#         f.write(f"Board shape: {sample['board'].shape}\n")
#         f.write(f"Move index:  {sample['move'].item()}\n")
#         f.write(f"Mask sum:    {sample['mask'].sum().item()} legal moves\n")

# print("Done! Check shard_000_summary.txt")


import torch
from model.chess_model import ChessModel

model, _, _, shard_id = ChessModel.load_checkpoint("outputs/models/latest.pt", device="cpu")
model.eval()

print(f"Model trained up to shard: {shard_id}")
print(f"\nModel architecture:\n{model}")
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
# ```

# Output will look like:
# ```
# Loaded checkpoint: shard 27, epoch 1

# Model trained up to shard: 27

# Model architecture:
# ChessModel(
#   (encoder): CNNEncoder(...)
#   (policy_head): PolicyHead(...)
#   (value_head): ValueHead(...)
# )

# Total parameters: 8,357,000