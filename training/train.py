# train.py - Complete version with unified endgame training
import sys
import os
import gc
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
import signal
import random

from model.chess_model import ChessModel
from training.loss import ChessLoss
from training.endgame_trainer import UnifiedTrainer

# ─────────────────────────────────────────
# Config - Default values (will be overridden by config file if exists)
# ─────────────────────────────────────────
try:
    from config.training_config import REGULAR_CONFIG, ENDGAME_CONFIG, COMMON_CONFIG
    print("✓ Loaded config from config/training_config.py")
except ImportError:
    print("⚠️ Config file not found, using defaults...")
    REGULAR_CONFIG = {
        "shard_dir": "data/shards",
        "output_dir": "outputs/models",
        "batch_size": 64,
        "accumulation_steps": 4,
        "epochs_per_shard": 2,
        "passes": 5,
        "learning_rate": 1e-3,
        "value_weight": 0.05,
        "label_smoothing": 0.1,
        "grad_clip": 1.0,
    }
    ENDGAME_CONFIG = {
        "enabled": True,
        "endgame_shard_dir": "data/endgame_shards",
        "endgame_weight": 2.0,
        "batch_ratio": 0.3,
        "epochs": 5,
        "learning_rate": 1e-4,
        "accumulation_steps": 4,
        "value_weight": 0.05,
        "label_smoothing": 0.1,
        "num_workers": 0,
    }
    COMMON_CONFIG = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "input_channels": 20,
        "num_moves": 4672,
        "num_workers": 2,
    }

SHARD_DIR          = REGULAR_CONFIG["shard_dir"]
OUTPUT_DIR         = REGULAR_CONFIG["output_dir"]
LOG_DIR            = "outputs/logs"
PLOT_DIR           = "outputs/plot"
METRICS_FILE       = f"{OUTPUT_DIR}/training_metrics.json"

BATCH_SIZE         = REGULAR_CONFIG["batch_size"]
ACCUMULATION_STEPS = REGULAR_CONFIG["accumulation_steps"]
EPOCHS_PER_SHARD   = REGULAR_CONFIG["epochs_per_shard"]
LR                 = REGULAR_CONFIG["learning_rate"]
VALUE_WEIGHT       = REGULAR_CONFIG["value_weight"]
LABEL_SMOOTHING    = REGULAR_CONFIG["label_smoothing"]
USE_FOCAL_LOSS     = False
GRAD_CLIP          = REGULAR_CONFIG["grad_clip"]
NUM_WORKERS        = COMMON_CONFIG.get("num_workers", 2)
PREFETCH_FACTOR    = 2
MAX_HISTORY        = 5000
SAVE_PER_SHARD     = False

PASSES             = REGULAR_CONFIG["passes"]
DEVICE             = COMMON_CONFIG["device"]
INPUT_CHANNELS     = COMMON_CONFIG["input_channels"]
NUM_MOVES          = COMMON_CONFIG["num_moves"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)
os.makedirs(PLOT_DIR,   exist_ok=True)

# Global flag for graceful shutdown
stop_training = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global stop_training
    print("\n\n⚠️ Received interrupt signal. Saving checkpoint and exiting gracefully...")
    stop_training = True

signal.signal(signal.SIGINT, signal_handler)

# ─────────────────────────────────────────
# Logger
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            f"{LOG_DIR}/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────
class ShardDataset(Dataset):
    def __init__(self, shard_path, augment=False):
        self.data    = torch.load(shard_path, weights_only=False)
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        board  = sample["board"]
        move   = sample["move"]
        mask   = sample["mask"]
        value  = sample.get("value", torch.tensor(0.0))
        return board, move, mask, value


# ─────────────────────────────────────────
# Loss plot functions
# ─────────────────────────────────────────
_fig  = None
_axes = None

def save_loss_plot(shard_ids, metrics_history):
    global _fig, _axes
    if _fig is None:
        _fig, _axes = plt.subplots(2, 3, figsize=(15, 10), facecolor="#0f0f0f")

    _fig.suptitle("Training Metrics", color="white", fontsize=14)
    configs = [
        (_axes[0, 0], metrics_history.get("policy_loss", []),         "#4A90D9", "Policy Loss"),
        (_axes[0, 1], metrics_history.get("value_loss", []),           "#E25C5C", "Value Loss"),
        (_axes[0, 2], metrics_history.get("total_loss", []),           "#7BC67E", "Total Loss"),
        (_axes[1, 0], metrics_history.get("move_accuracy", []),        "#FFD966", "Move Accuracy"),
        (_axes[1, 1], metrics_history.get("value_sign_accuracy", []),  "#9B59B6", "Value Sign Acc"),
        (_axes[1, 2], metrics_history.get("value_mae", []),            "#E67E22", "Value MAE"),
    ]
    for ax, values, color, title in configs:
        ax.clear()
        ax.set_facecolor("#1a1a1a")
        if values:
            ax.plot(shard_ids, values, color=color, linewidth=1.0, alpha=0.5, marker=".")
            if len(values) >= 7:
                smoothed = np.convolve(values, [1/7]*7, mode="same")
                ax.plot(shard_ids, smoothed, color=color, linewidth=2.0)
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xlabel("Shard", color="#cccccc")
        ax.tick_params(colors="#cccccc")
        ax.grid(color="#2a2a2a", linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    _fig.tight_layout()
    _fig.savefig(
        f"{PLOT_DIR}/training_metrics.png", dpi=150,
        bbox_inches="tight", facecolor="#0f0f0f"
    )
    plt.close(_fig)
    _fig = None
    _axes = None


def save_metrics_json(shard_ids, metrics_history, total_epochs_completed):
    data = {
        "shard_ids": shard_ids,
        "metrics": metrics_history,
        "total_epochs_completed": total_epochs_completed,
        "last_updated": datetime.now().isoformat()
    }
    with open(METRICS_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def load_metrics_json():
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, 'r') as f:
                data = json.load(f)
                return data.get("shard_ids", []), data.get("metrics", {})
        except:
            return [], {}
    return [], {}


# ─────────────────────────────────────────
# Save checkpoint
# ─────────────────────────────────────────
def save_checkpoint(path, model, optimizer, scheduler, shard_id, epoch, metrics, 
                    total_epochs_completed=None, current_pass=None):
    tmp_path = path + ".tmp"
    
    checkpoint_data = {
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "shard_id":        shard_id,
        "epoch":           epoch,
        "metrics":         metrics,
    }
    
    if scheduler is not None:
        try:
            checkpoint_data["scheduler_state"] = scheduler.state_dict()
        except:
            pass
    
    if total_epochs_completed is not None:
        checkpoint_data["total_epochs_completed"] = total_epochs_completed
    
    if current_pass is not None:
        checkpoint_data["current_pass"] = current_pass
    
    torch.save(checkpoint_data, tmp_path)
    os.replace(tmp_path, path)


# ─────────────────────────────────────────
# Train one shard
# ─────────────────────────────────────────
def train_shard(model, optimizer, scheduler, loss_fn, shard_path, shard_id, epoch=0):
    dataset = ShardDataset(shard_path, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        persistent_workers=False,
    )

    model.train()
    total_p, total_v, total_t = 0.0, 0.0, 0.0
    total_acc, total_sign, total_mae = 0.0, 0.0, 0.0
    n = 0

    pbar = tqdm(loader, desc=f"Shard {shard_id:03d} Ep{epoch}", unit="batch", leave=False)
    optimizer.zero_grad()

    for batch_idx, (boards, moves, masks, values) in enumerate(pbar):
        boards = boards.to(DEVICE, non_blocking=True)
        moves  = moves.to(DEVICE,  non_blocking=True)
        masks  = masks.to(DEVICE,  non_blocking=True)
        values = values.to(DEVICE, non_blocking=True)

        policy_logits, value_pred = model(boards)

        loss, p_loss, v_loss, acc, sign_acc, mae = loss_fn(
            policy_logits, moves, value_pred, values, masks
        )

        (loss / ACCUMULATION_STEPS).backward()

        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            
            if scheduler is not None:
                try:
                    scheduler.step()
                except Exception:
                    pass

        total_p    += p_loss.item()
        total_v    += v_loss.item()
        total_t    += loss.item()
        total_acc  += acc.item()
        total_sign += sign_acc.item()
        total_mae  += mae.item()
        n          += 1

        current_lr = 0.0
        if scheduler is not None:
            try:
                current_lr = scheduler.get_last_lr()[0]
            except:
                pass
        
        pbar.set_postfix(
            p=f"{p_loss.item():.3f}",
            v=f"{v_loss.item():.3f}",
            acc=f"{acc.item():.3f}",
            lr=f"{current_lr:.2e}"
        )

    del loader, dataset
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    metrics = {
        "policy_loss":         total_p / n,
        "value_loss":          total_v / n,
        "total_loss":          total_t / n,
        "move_accuracy":       total_acc / n,
        "value_sign_accuracy": total_sign / n,
        "value_mae":           total_mae / n,
    }

    log.info(
        f"Shard {shard_id:03d} Ep{epoch} | "
        f"Policy: {metrics['policy_loss']:.4f}  "
        f"Value: {metrics['value_loss']:.4f}  "
        f"Acc: {metrics['move_accuracy']:.3f}  "
        f"Sign: {metrics['value_sign_accuracy']:.3f}  "
    )
    return metrics


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    global stop_training
    
    checkpoint_path = f"{OUTPUT_DIR}/latest.pt"
    
    # Load all shards
    all_shards = sorted(glob.glob(f"{SHARD_DIR}/shard_*.pt"))
    if not all_shards:
        log.error(f"No shards found in {SHARD_DIR}")
        return
    
    # Calculate total epochs target
    total_shards_count = len(all_shards)
    TOTAL_EPOCHS_TARGET = total_shards_count * EPOCHS_PER_SHARD * PASSES
    
    log.info("="*60)
    log.info("REGULAR TRAINING PHASE")
    log.info("="*60)
    log.info(f"Total shards: {total_shards_count}")
    log.info(f"Passes: {PASSES} | Epochs per shard: {EPOCHS_PER_SHARD}")
    log.info(f"Target total epochs: {TOTAL_EPOCHS_TARGET}")
    log.info("="*60)
    
    # Initialize model and optimizer
    model = ChessModel(input_channels=INPUT_CHANNELS, num_moves=NUM_MOVES).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # Check if we already completed regular training
    regular_completed = False
    start_shard_idx = 0
    start_epoch_in_shard = 0
    total_epochs_completed = 0
    current_pass = 1
    
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            
            last_shard_id = ckpt.get("shard_id", -1)
            last_epoch = ckpt.get("epoch", -1)
            total_epochs_completed = ckpt.get("total_epochs_completed", 0)
            current_pass = ckpt.get("current_pass", 1)
            
            log.info(f"Resuming from shard {last_shard_id}, epoch {last_epoch}")
            log.info(f"Total epochs completed: {total_epochs_completed}")
            
            # Check if regular training is complete
            if total_epochs_completed >= TOTAL_EPOCHS_TARGET:
                regular_completed = True
                log.info("Regular training already completed!")
            
            # Find resume position
            for idx, shard_path in enumerate(all_shards):
                shard_num = int(os.path.basename(shard_path).split("_")[1].split(".")[0])
                if shard_num == last_shard_id:
                    start_shard_idx = idx
                    start_epoch_in_shard = last_epoch + 1
                    break
            
            if start_epoch_in_shard >= EPOCHS_PER_SHARD:
                start_shard_idx += 1
                start_epoch_in_shard = 0
            
            del ckpt
            gc.collect()
            
        except Exception as e:
            log.warning(f"Checkpoint load failed: {e}")
    
    # ── Regular Training Phase ────────────────────────────────────────────────
    if not regular_completed and total_epochs_completed < TOTAL_EPOCHS_TARGET:
        # Calculate steps for scheduler
        try:
            sample_shard = ShardDataset(all_shards[0])
            samples_per_shard = len(sample_shard)
            steps_per_shard = max(1, samples_per_shard // BATCH_SIZE // ACCUMULATION_STEPS)
            steps_per_epoch = steps_per_shard * len(all_shards)
            remaining_epochs = TOTAL_EPOCHS_TARGET - total_epochs_completed
            total_steps_needed = remaining_epochs * steps_per_epoch
        except:
            total_steps_needed = remaining_epochs * len(all_shards) * 100
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=LR,
            total_steps=total_steps_needed,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=10000.0,
        )
        
        loss_fn = ChessLoss(
            value_weight=VALUE_WEIGHT,
            label_smoothing=LABEL_SMOOTHING,
            use_focal_loss=USE_FOCAL_LOSS,
        )
        
        shard_ids, metrics_history = load_metrics_json()
        if not metrics_history:
            metrics_history = {
                "policy_loss": [], "value_loss": [], "total_loss": [],
                "move_accuracy": [], "value_sign_accuracy": [], "value_mae": []
            }
        
        overall_pbar = tqdm(total=TOTAL_EPOCHS_TARGET, desc="Regular Training", position=0, leave=True)
        overall_pbar.update(total_epochs_completed)
        
        # Training loop
        current_shard_idx = start_shard_idx
        current_epoch_in_shard = start_epoch_in_shard
        
        while total_epochs_completed < TOTAL_EPOCHS_TARGET and not stop_training:
            for shard_idx in range(current_shard_idx, len(all_shards)):
                if stop_training:
                    break
                    
                shard_path = all_shards[shard_idx]
                shard_id = int(os.path.basename(shard_path).split("_")[1].split(".")[0])
                
                start_epoch = current_epoch_in_shard if shard_idx == current_shard_idx else 0
                
                for epoch in range(start_epoch, EPOCHS_PER_SHARD):
                    if stop_training or total_epochs_completed >= TOTAL_EPOCHS_TARGET:
                        break
                    
                    metrics = train_shard(
                        model, optimizer, scheduler, loss_fn,
                        shard_path, shard_id, epoch
                    )
                    
                    total_epochs_completed += 1
                    overall_pbar.update(1)
                    overall_pbar.set_postfix(acc=f"{metrics['move_accuracy']:.3f}")
                    
                    shard_ids.append(shard_id)
                    for key in metrics_history:
                        metrics_history[key].append(metrics.get(key, 0.0))
                    
                    if len(shard_ids) > MAX_HISTORY:
                        shard_ids = shard_ids[-MAX_HISTORY:]
                        for k in metrics_history:
                            metrics_history[k] = metrics_history[k][-MAX_HISTORY:]
                    
                    save_checkpoint(
                        checkpoint_path, model, optimizer, scheduler,
                        shard_id, epoch, metrics, total_epochs_completed, current_pass
                    )
                    save_metrics_json(shard_ids, metrics_history, total_epochs_completed)
                    
                    if total_epochs_completed % 10 == 0:
                        save_loss_plot(shard_ids, metrics_history)
                
                current_epoch_in_shard = 0
            
            if total_epochs_completed < TOTAL_EPOCHS_TARGET and not stop_training:
                current_pass += 1
                current_shard_idx = 0
                overall_pbar.set_description(f"Regular Training (Pass {current_pass}/{PASSES})")
                log.info(f"Starting pass {current_pass}/{PASSES}")
        
        overall_pbar.close()
        save_loss_plot(shard_ids, metrics_history)
        
        log.info("="*60)
        log.info(f"Regular training complete! {total_epochs_completed}/{TOTAL_EPOCHS_TARGET} epochs")
        log.info("="*60)
        
        # Save regular training final checkpoint
        regular_final_path = f"{OUTPUT_DIR}/regular_trained_final.pt"
        torch.save({
            "model_state": model.state_dict(),
            "regular_epochs": total_epochs_completed,
            "regular_metrics": metrics_history,
            "timestamp": datetime.now().isoformat()
        }, regular_final_path)
        log.info(f"Regular training final model saved to {regular_final_path}")
    
    # ── UNIFIED ENDGAME TRAINING PHASE (Continues on same model) ──────────────
       # ── UNIFIED TRAINING (Combine Regular + Endgame) ─────────────────────────
    if ENDGAME_CONFIG["enabled"] and not stop_training:
        log.info("\n" + "="*60)
        log.info("UNIFIED TRAINING - COMBINING REGULAR + ENDGAME DATA")
        log.info("="*60)
    
        # Check if endgame shards exist
        endgame_shards = glob.glob(f"{ENDGAME_CONFIG['endgame_shard_dir']}/endgame_shard_*.pt")
        if not endgame_shards:
            log.warning("No endgame shards found! Skipping unified training.")
            log.info("Run: python preprocessing/generate_endgame.py to generate endgame dataset")
        else:
            # Get all regular shard paths
            regular_shard_paths = all_shards
            
            log.info(f"Regular shards: {len(regular_shard_paths)}")
            log.info(f"Endgame shards: {len(endgame_shards)}")
            log.info(f"Training will combine both datasets into one unified model")
            
            # Import UnifiedTrainer
            from training.endgame_trainer import UnifiedTrainer
            
            # Initialize unified trainer
            unified_trainer = UnifiedTrainer(model, ENDGAME_CONFIG, checkpoint_path=checkpoint_path)
            
            # Check if we already started
            start_epoch = 0
            if os.path.exists(checkpoint_path):
                try:
                    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
                    if "unified_epoch" in ckpt:
                        start_epoch = ckpt.get("unified_epoch", -1) + 1
                        log.info(f"Resuming unified training from epoch {start_epoch}")
                except:
                    pass
            
            # Run unified training
            unified_metrics = unified_trainer.run_unified_training(
                regular_shard_paths, 
                start_epoch=start_epoch
            )
            
            if unified_metrics:
                log.info("\n" + "="*60)
                log.info("UNIFIED TRAINING SUMMARY")
                log.info("="*60)
                for i, metrics in enumerate(unified_metrics):
                    log.info(f"Epoch {i}: Acc={metrics['move_accuracy']:.3f}, Loss={metrics['total_loss']:.3f}")
                log.info("="*60)
                log.info(f"✓ UNIFIED MODEL SAVED to {checkpoint_path}")
                log.info("  This model now knows regular positions AND endgames!")
    
    # ── Final Summary ─────────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("TRAINING PIPELINE COMPLETE!")
    log.info("="*60)
    log.info(f"Regular training epochs: {total_epochs_completed}")
    
    # Safely check endgame metrics
    if ENDGAME_CONFIG["enabled"]:
        if 'unified_metrics' in locals() and unified_metrics:
            log.info(f"Endgame training: Completed ({len(unified_metrics)} unified epochs)")
        else:
            log.info(f"Endgame training: Skipped/No data")
    else:
        log.info(f"Endgame training: Disabled")
    
    log.info(f"Final model saved to: {checkpoint_path}")
    
    # Verify the final model exists
    if os.path.exists(checkpoint_path):
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
        log.info(f"✓ Model file size: {file_size:.2f} MB")
    log.info("="*60)


if __name__ == "__main__":
    main()