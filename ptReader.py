# ptReader.py - Fixed version
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
from model.chess_model import ChessModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def read_checkpoint():
    """Read and display checkpoint info"""
    
    print("="*60)
    print("MODEL STATISTICS")
    print("="*60)
    
    # Load checkpoint
    checkpoint = torch.load("outputs/models/latest.pt", map_location='cpu')
    
    # File info
    size_mb = os.path.getsize("outputs/models/latest.pt") / (1024 * 1024)
    print(f"\n📁 File size: {size_mb:.2f} MB")
    
    # Checkpoint info
    if 'shard_id' in checkpoint:
        print(f"🎯 Shard ID: {checkpoint['shard_id']}")
    if 'epoch' in checkpoint:
        print(f"🎯 Epoch: {checkpoint['epoch']}")
    if 'step' in checkpoint:
        print(f"🎯 Step: {checkpoint['step']}")
    
    # Load model properly
    print("\n🔄 Loading model...")
    model, optimizer, scheduler, shard_id = ChessModel.load_checkpoint(
        "outputs/models/latest.pt", 
        device=DEVICE
    )
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Info:")
    print(f"   Shard ID: {shard_id}")
    print(f"   Device: {DEVICE}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model class: {model.__class__.__name__}")
    
    # Test with a sample position
    import chess
    from utils.fen_utils import fen_to_tensor
    from utils.generate_move_mask import generate_move_mask
    from utils.move_index_encoding import policy_index_to_move
    
    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board_tensor = torch.tensor(fen_to_tensor(test_fen), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    mask = torch.tensor(generate_move_mask(test_fen), dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        policy_logits, value = model(board_tensor)
        policy_logits = policy_logits.squeeze(0) + (mask - 1) * 1e9
        best_idx = torch.argmax(policy_logits).item()
        best_move = policy_index_to_move(best_idx)
    
    print(f"\n🧪 Test prediction:")
    print(f"   Best move: {best_move.uci()}")
    print(f"   Value: {value.item():.3f}")
    
    return model

if __name__ == "__main__":
    read_checkpoint()