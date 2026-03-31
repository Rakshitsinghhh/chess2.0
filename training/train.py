import sys
import os
import gc
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
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

from model.chess_model import ChessModel
from training.loss import ChessLoss

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
SHARD_DIR          = "data/shards"
OUTPUT_DIR         = "outputs/models"
LOG_DIR            = "outputs/logs"
PLOT_DIR           = "outputs/plot"
METRICS_FILE       = f"{OUTPUT_DIR}/training_metrics.json"

BATCH_SIZE         = 64
ACCUMULATION_STEPS = 4        # effective batch = 256
EPOCHS_PER_SHARD   = 2         # How many epochs to train on each shard per pass
LR                 = 1e-3
VALUE_WEIGHT       = 0.05
LABEL_SMOOTHING    = 0.1
USE_FOCAL_LOSS     = False
GRAD_CLIP          = 1.0
NUM_WORKERS        = 2
PREFETCH_FACTOR    = 2
MAX_HISTORY        = 5000      # Increased to store all metrics
SAVE_PER_SHARD     = False

# TRAINING CONFIGURATION
PASSES = 5  # Set number of complete passes through all shards
# TOTAL_EPOCHS_TARGET will be calculated automatically based on PASSES

DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_CHANNELS     = 20
NUM_MOVES          = 4672

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

# Register signal handler
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
# Loss plot
# ─────────────────────────────────────────
_fig  = None
_axes = None

def save_loss_plot(shard_ids, metrics_history):
    """Save training metrics plot"""
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
    """Save metrics to JSON for later analysis"""
    data = {
        "shard_ids": shard_ids,
        "metrics": metrics_history,
        "total_epochs_completed": total_epochs_completed,
        "last_updated": datetime.now().isoformat()
    }
    with open(METRICS_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def load_metrics_json():
    """Load previous metrics if they exist"""
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, 'r') as f:
                data = json.load(f)
                return data.get("shard_ids", []), data.get("metrics", {})
        except:
            return [], {}
    return [], {}


# ─────────────────────────────────────────
# Save checkpoint — guaranteed atomic write
# ─────────────────────────────────────────
def save_checkpoint(path, model, optimizer, scheduler, shard_id, epoch, metrics, total_epochs_completed=None, current_pass=None):
    """Save checkpoint atomically"""
    tmp_path = path + ".tmp"
    
    checkpoint_data = {
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "shard_id":        shard_id,
        "epoch":           epoch,
        "metrics":         metrics,
    }
    
    # Save scheduler state only if it exists and has steps left
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
def train_shard(model, optimizer, scheduler, loss_fn, shard_path, shard_id, epoch=0, steps_taken=0):
    """Train on a single shard for one epoch"""
    dataset = ShardDataset(shard_path, augment=False)
    loader  = DataLoader(
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
    local_steps = 0

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
            
            # Step scheduler if it exists
            if scheduler is not None:
                try:
                    scheduler.step()
                    local_steps += 1
                except Exception as e:
                    # If scheduler fails, continue without it
                    pass

        total_p    += p_loss.item()
        total_v    += v_loss.item()
        total_t    += loss.item()
        total_acc  += acc.item()
        total_sign += sign_acc.item()
        total_mae  += mae.item()
        n          += 1

        # Get current LR safely
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
    
    # Calculate total epochs target based on passes
    total_shards_count = len(all_shards)
    TOTAL_EPOCHS_TARGET = total_shards_count * EPOCHS_PER_SHARD * PASSES
    log.info(f"Training configuration: {PASSES} pass(es) through {total_shards_count} shards")
    log.info(f"Each shard gets {EPOCHS_PER_SHARD} epochs per pass")
    log.info(f"Total target epochs: {TOTAL_EPOCHS_TARGET}")
    
    # Initialize variables
    start_shard_idx = 0
    start_epoch_in_shard = 0
    total_epochs_completed = 0
    current_pass = 1
    model = ChessModel(input_channels=INPUT_CHANNELS, num_moves=NUM_MOVES).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # Load previous metrics if they exist
    shard_ids, metrics_history = load_metrics_json()
    
    # ── Load checkpoint if it exists ──────────────────────────────────────────
    if os.path.exists(checkpoint_path):
        try:
            log.info(f"Loading checkpoint from {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            
            # Get checkpoint info
            last_shard_id = ckpt.get("shard_id", -1)
            last_epoch = ckpt.get("epoch", -1)
            total_epochs_completed = ckpt.get("total_epochs_completed", 0)
            current_pass = ckpt.get("current_pass", 1)
            
            log.info(f"Checkpoint info: shard={last_shard_id}, epoch={last_epoch}, total_epochs={total_epochs_completed}, pass={current_pass}")
            
            # Find where to resume
            found = False
            for idx, shard_path in enumerate(all_shards):
                shard_num = int(os.path.basename(shard_path).split("_")[1].split(".")[0])
                if shard_num == last_shard_id:
                    start_shard_idx = idx
                    start_epoch_in_shard = last_epoch + 1
                    found = True
                    break
            
            if not found:
                log.warning(f"Shard {last_shard_id} not found, starting from beginning of current pass")
                start_shard_idx = 0
                start_epoch_in_shard = 0
            
            # If we completed all epochs for this shard, move to next
            if start_epoch_in_shard >= EPOCHS_PER_SHARD:
                start_shard_idx += 1
                start_epoch_in_shard = 0
                log.info(f"Completed all epochs for shard {last_shard_id}, moving to next shard")
            
            del ckpt
            gc.collect()
            
        except Exception as e:
            log.warning(f"⚠ Checkpoint load failed: {e} — starting fresh")
            start_shard_idx = 0
            start_epoch_in_shard = 0
            total_epochs_completed = 0
            current_pass = 1
            shard_ids = []
            metrics_history = {}
    else:
        log.info("No checkpoint found — starting fresh training")
    
    # ── Check if we've reached target ─────────────────────────────────────────
    if total_epochs_completed >= TOTAL_EPOCHS_TARGET:
        log.info(f"✓ Target of {TOTAL_EPOCHS_TARGET} epochs already completed!")
        log.info(f"  Total epochs completed: {total_epochs_completed}")
        log.info(f"  Completed {PASSES} complete pass(es) through all data")
        return
    
    # ── Calculate remaining training and create NEW scheduler ─────────────────
    remaining_epochs = TOTAL_EPOCHS_TARGET - total_epochs_completed
    
    # Calculate total steps needed for remaining training
    try:
        sample_shard = ShardDataset(all_shards[0])
        samples_per_shard = len(sample_shard)
        steps_per_shard = max(1, samples_per_shard // BATCH_SIZE // ACCUMULATION_STEPS)
        steps_per_epoch = steps_per_shard * len(all_shards)
        total_steps_needed = remaining_epochs * steps_per_epoch
        log.info(f"Calculated steps: {steps_per_shard} steps/shard, {steps_per_epoch} steps/epoch, {total_steps_needed} total steps needed")
    except Exception as e:
        log.warning(f"Could not calculate exact steps: {e}, using estimate")
        total_steps_needed = remaining_epochs * len(all_shards) * 100
    
    # ALWAYS create a new scheduler for the remaining training
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LR,
        total_steps=total_steps_needed,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=10000.0,
    )
    log.info(f"✓ Created NEW scheduler for {remaining_epochs} remaining epochs ({total_steps_needed} steps)")
    
    # Initialize metrics history if empty
    if not metrics_history:
        metrics_history = {
            "policy_loss": [], "value_loss": [], "total_loss": [],
            "move_accuracy": [], "value_sign_accuracy": [], "value_mae": []
        }
    
    # ── Log training info ─────────────────────────────────────────────────────
    log.info("="*60)
    log.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    log.info(f"Total shards: {len(all_shards)}")
    log.info(f"Passes: {PASSES}  |  Current pass: {current_pass}")
    log.info(f"Target total epochs: {TOTAL_EPOCHS_TARGET}")
    log.info(f"Epochs completed: {total_epochs_completed}")
    log.info(f"Remaining epochs: {remaining_epochs}")
    log.info(f"EPOCHS_PER_SHARD: {EPOCHS_PER_SHARD}")
    log.info(f"Device: {DEVICE}  |  Batch: {BATCH_SIZE} (eff. {BATCH_SIZE * ACCUMULATION_STEPS})")
    log.info(f"Resuming from shard {start_shard_idx} (shard {start_shard_idx+1}/{len(all_shards)}), epoch {start_epoch_in_shard}")
    log.info("="*60)
    log.info("⚠️ Press Ctrl+C to save checkpoint and exit gracefully")
    log.info("="*60)
    
    # ── Loss function ─────────────────────────────────────────────────────────
    loss_fn = ChessLoss(
        value_weight=VALUE_WEIGHT,
        label_smoothing=LABEL_SMOOTHING,
        use_focal_loss=USE_FOCAL_LOSS,
    )
    
    # Create overall progress bar
    overall_pbar = tqdm(total=TOTAL_EPOCHS_TARGET, desc=f"Overall Progress (Pass {current_pass}/{PASSES})", position=0, leave=True)
    overall_pbar.update(total_epochs_completed)  # Show already completed epochs
    
    # ── Training loop with multi-pass support ─────────────────────────────────
    epochs_trained_this_run = 0
    current_shard_idx = start_shard_idx
    current_epoch_in_shard = start_epoch_in_shard
    
    # Main training loop - continues until target reached or stop flag
    while total_epochs_completed < TOTAL_EPOCHS_TARGET and not stop_training:
        # Loop through shards
        for shard_idx in range(current_shard_idx, len(all_shards)):
            if stop_training:
                break
                
            shard_path = all_shards[shard_idx]
            shard_id = int(os.path.basename(shard_path).split("_")[1].split(".")[0])
            
            # Determine starting epoch for this shard
            start_epoch = current_epoch_in_shard if shard_idx == current_shard_idx else 0
            
            # Train epochs for this shard
            for epoch in range(start_epoch, EPOCHS_PER_SHARD):
                if stop_training or total_epochs_completed >= TOTAL_EPOCHS_TARGET:
                    break
                
                # Train on this shard
                metrics = train_shard(
                    model, optimizer, scheduler, loss_fn,
                    shard_path, shard_id, epoch
                )
                
                total_epochs_completed += 1
                epochs_trained_this_run += 1
                
                # Update overall progress bar
                overall_pbar.update(1)
                overall_pbar.set_postfix(
                    acc=f"{metrics['move_accuracy']:.3f}",
                    loss=f"{metrics['total_loss']:.3f}",
                    shard=f"{shard_id}",
                    pass_num=f"{current_pass}"
                )
                
                # Store metrics
                shard_ids.append(shard_id)
                for key in metrics_history:
                    metrics_history[key].append(metrics.get(key, 0.0))
                
                # Trim history if needed
                if len(shard_ids) > MAX_HISTORY:
                    shard_ids = shard_ids[-MAX_HISTORY:]
                    for k in metrics_history:
                        metrics_history[k] = metrics_history[k][-MAX_HISTORY:]
                
                # Save checkpoint after every epoch
                save_checkpoint(
                    checkpoint_path,
                    model, optimizer, scheduler,
                    shard_id, epoch, metrics,
                    total_epochs_completed,
                    current_pass
                )
                
                # Save metrics JSON
                save_metrics_json(shard_ids, metrics_history, total_epochs_completed)
                
                # Save plot periodically (every 10 epochs to save time)
                if total_epochs_completed % 10 == 0:
                    save_loss_plot(shard_ids, metrics_history)
                
                # Calculate remaining for logging
                remaining = TOTAL_EPOCHS_TARGET - total_epochs_completed
                log.info(f"✓ Checkpoint saved | Progress: {total_epochs_completed}/{TOTAL_EPOCHS_TARGET} epochs | Remaining: {remaining} | Pass: {current_pass}/{PASSES} | Acc: {metrics['move_accuracy']:.3f}")
            
            # Reset for next shard
            current_epoch_in_shard = 0
            
            # Check if we reached target
            if total_epochs_completed >= TOTAL_EPOCHS_TARGET or stop_training:
                break
        
        # After completing all shards, check if we need another pass
        if total_epochs_completed < TOTAL_EPOCHS_TARGET and not stop_training:
            current_pass += 1
            current_shard_idx = 0
            current_epoch_in_shard = 0
            overall_pbar.set_description(f"Overall Progress (Pass {current_pass}/{PASSES})")
            log.info(f"🎉 Completed pass {current_pass-1}/{PASSES}! Starting pass {current_pass}/{PASSES}")
            log.info(f"   Progress: {total_epochs_completed}/{TOTAL_EPOCHS_TARGET} epochs completed")
        else:
            break
    
    # Close progress bar
    overall_pbar.close()
    
    # Save final checkpoint if stopped by user
    if stop_training:
        log.info("⚠️ Training stopped by user. Final checkpoint saved.")
        # Save one more checkpoint with current state
        save_checkpoint(
            checkpoint_path,
            model, optimizer, scheduler,
            shard_id, epoch, metrics,
            total_epochs_completed,
            current_pass
        )
    
    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("="*60)
    if total_epochs_completed >= TOTAL_EPOCHS_TARGET:
        log.info(f"✓ Training complete! Reached target: {total_epochs_completed}/{TOTAL_EPOCHS_TARGET} epochs")
        log.info(f"✓ Completed {PASSES} complete pass(es) through all {total_shards_count} shards")
    else:
        log.info(f"⚠ Training stopped: {total_epochs_completed}/{TOTAL_EPOCHS_TARGET} epochs")
        log.info(f"   Completed {current_pass-1} full passes, started pass {current_pass}")
    log.info(f"Epochs trained in this run: {epochs_trained_this_run}")
    if metrics_history and metrics_history.get("move_accuracy"):
        log.info(f"Final move accuracy: {metrics_history['move_accuracy'][-1]:.4f}")
        log.info(f"Final policy loss: {metrics_history['policy_loss'][-1]:.4f}")
        log.info(f"Final value loss: {metrics_history['value_loss'][-1]:.4f}")
    log.info(f"Checkpoint saved at: {checkpoint_path}")
    log.info("="*60)


if __name__ == "__main__":
    main()