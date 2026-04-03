# run_unified_only.py
"""Run only unified training (skip regular training)"""
import sys
import os
import torch
import logging
from datetime import datetime

sys.path.append(os.path.dirname(__file__))

from model.chess_model import ChessModel
from training.endgame_trainer import UnifiedTrainer
from config.training_config import ENDGAME_CONFIG, COMMON_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"outputs/logs/unified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

def main():
    print("\n" + "="*60)
    print("UNIFIED TRAINING ONLY")
    print("="*60)
    
    # Check if model exists
    model_path = "outputs/models/latest.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train a regular model first or provide a model file")
        return
    
    # Load existing model
    print(f"\n📁 Loading model from: {model_path}")
    device = COMMON_CONFIG["device"]
    model = ChessModel(
        input_channels=COMMON_CONFIG["input_channels"],
        num_moves=COMMON_CONFIG["num_moves"]
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"✅ Model loaded successfully")
    
    # Load regular shards
    regular_shards = sorted(glob.glob("data/shards/shard_*.pt"))
    if not regular_shards:
        print("❌ No regular shards found!")
        return
    
    print(f"📊 Regular shards: {len(regular_shards)}")
    
    # Check endgame shards
    endgame_shards = glob.glob(f"{ENDGAME_CONFIG['endgame_shard_dir']}/endgame_shard_*.pt")
    if not endgame_shards:
        print("❌ No endgame shards found!")
        print("Run: python preprocessing/generate_endgame.py")
        return
    
    print(f"📊 Endgame shards: {len(endgame_shards)}")
    
    # Initialize unified trainer
    unified_trainer = UnifiedTrainer(model, ENDGAME_CONFIG, checkpoint_path=model_path)
    
    # Check if we already started unified training
    start_epoch = 0
    if "unified_epoch" in checkpoint:
        start_epoch = checkpoint.get("unified_epoch", -1) + 1
        print(f"🔄 Resuming unified training from epoch {start_epoch}")
    
    # Run unified training
    print("\n" + "="*60)
    print("🚀 STARTING UNIFIED TRAINING")
    print("="*60)
    
    unified_metrics = unified_trainer.run_unified_training(
        regular_shards,
        start_epoch=start_epoch
    )
    
    if unified_metrics:
        print("\n" + "="*60)
        print("✅ UNIFIED TRAINING COMPLETE!")
        print("="*60)
        print(f"Epochs completed: {len(unified_metrics)}")
        print(f"Final accuracy: {unified_metrics[-1]['move_accuracy']:.3f}")
        print(f"Final loss: {unified_metrics[-1]['total_loss']:.3f}")
        print(f"Model saved to: {model_path}")
    else:
        print("❌ Unified training produced no results")

if __name__ == "__main__":
    main()