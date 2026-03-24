import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
import glob
import matplotlib.pyplot as plt

from model.chess_model import ChessModel

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
SHARD_DIR   = "data/shards"
OUTPUT_DIR  = "outputs/models"
LOG_DIR     = "outputs/logs"
PLOT_DIR    = "outputs/plot"
BATCH_SIZE  = 256
EPOCHS      = 1
LR          = 1e-3
VALUE_WEIGHT = 0.05  
GRAD_CLIP    = 1.0    
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)
os.makedirs(PLOT_DIR,   exist_ok=True)

# ─────────────────────────────────────────
# Logger
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/training.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Dataset — now loads value target too
# ─────────────────────────────────────────
class ShardDataset(Dataset):
    def __init__(self, shard_path):
        self.data = torch.load(shard_path, weights_only=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        board  = sample["board"]
        move   = sample["move"]
        mask   = sample["mask"]
        # graceful fallback for old shards that don't have value yet
        value  = sample.get("value", torch.tensor(0.0))
        return board, move, mask, value


# ─────────────────────────────────────────
# Loss plot
# ─────────────────────────────────────────
def save_loss_plot(shard_ids, policy_losses, value_losses, total_losses):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="#0f0f0f")
    fig.suptitle("Training Loss", color="white", fontsize=13)

    pairs = [
        (axes[0], policy_losses, "#4A90D9", "Policy Loss"),
        (axes[1], value_losses,  "#E25C5C", "Value Loss"),
        (axes[2], total_losses,  "#7BC67E", "Total Loss"),
    ]

    import numpy as np
    for ax, losses, color, title in pairs:
        ax.set_facecolor("#1a1a1a")
        ax.plot(shard_ids, losses, color=color, linewidth=1.0, alpha=0.5)
        if len(losses) >= 7:
            smoothed = np.convolve(losses, [1/7]*7, mode="same")
            ax.plot(shard_ids, smoothed, color=color, linewidth=2.0)
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xlabel("Shard", color="#cccccc")
        ax.tick_params(colors="#cccccc")
        ax.grid(color="#2a2a2a", linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/loss_curve.png", dpi=150,
                bbox_inches="tight", facecolor="#0f0f0f")
    plt.close()


# ─────────────────────────────────────────
# Train one shard
# ─────────────────────────────────────────
def train_shard(model, optimizer, scheduler, shard_path, shard_id):
    dataset = ShardDataset(shard_path)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=2, pin_memory=(DEVICE == "cuda"))

    policy_loss_fn = nn.CrossEntropyLoss()

    model.train()
    total_policy_loss = 0.0
    total_value_loss  = 0.0
    total_loss_sum    = 0.0
    total_batches     = 0

    pbar = tqdm(loader, desc=f"Shard {shard_id:03d}", unit="batch")

    for boards, moves, masks, values in pbar:
        boards = boards.to(DEVICE)
        moves  = moves.to(DEVICE)
        masks  = masks.to(DEVICE)
        values = values.to(DEVICE)               # (B,) float in [-1, +1]

        optimizer.zero_grad()

        policy_logits, value_pred = model(boards)  # (B,4672), (B,1)

        # mask illegal moves
        policy_logits = policy_logits + (masks - 1) * 1e9

        # ── Policy loss ───────────────────────────────────────────────────────
        p_loss = policy_loss_fn(policy_logits, moves)

        # ── Value loss ────────────────────────────────────────────────────────
        v_loss = F.mse_loss(value_pred.squeeze(-1), values)

        # ── Total loss ────────────────────────────────────────────────────────
        loss = p_loss + VALUE_WEIGHT * v_loss

        loss.backward()

        # gradient clipping — prevents exploding gradients in deep residual net
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)

        optimizer.step()

        total_policy_loss += p_loss.item()
        total_value_loss  += v_loss.item()
        total_loss_sum    += loss.item()
        total_batches     += 1

        pbar.set_postfix(
            p=f"{p_loss.item():.3f}",
            v=f"{v_loss.item():.3f}",
            t=f"{loss.item():.3f}"
        )

    avg_p = total_policy_loss / total_batches
    avg_v = total_value_loss  / total_batches
    avg_t = total_loss_sum    / total_batches

    log.info(f"Shard {shard_id:03d} | Policy: {avg_p:.4f}  Value: {avg_v:.4f}  Total: {avg_t:.4f}  LR: {scheduler.get_last_lr()[0]:.6f}")
    return avg_p, avg_v, avg_t


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    checkpoint_path = f"{OUTPUT_DIR}/latest.pt"

    if os.path.exists(checkpoint_path):
        model, opt_state, last_epoch, last_shard = ChessModel.load_checkpoint(
            checkpoint_path, DEVICE
        )
        optimizer = optim.Adam(model.parameters(), lr=LR)
        if opt_state:
            optimizer.load_state_dict(opt_state)
        start_shard = last_shard + 1
        log.info(f"Resuming from shard {start_shard}")
    else:
        model       = ChessModel().to(DEVICE)
        optimizer   = optim.Adam(model.parameters(), lr=LR)
        start_shard = 0
        log.info("Starting fresh training")

    # LR scheduler — halve LR every 30 shards
    # e.g. shards 0–29 = 1e-3, 30–59 = 5e-4, 60–89 = 2.5e-4, 90+ = 1.25e-4
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.5, last_epoch=start_shard - 1
    )

    all_shards = sorted(glob.glob(f"{SHARD_DIR}/shard_*.pt"))
    log.info(f"Total shards found : {len(all_shards)}")
    log.info(f"Starting from shard: {start_shard}")
    log.info(f"Device             : {DEVICE}")
    log.info(f"Value loss weight  : {VALUE_WEIGHT}")
    log.info(f"Gradient clip      : {GRAD_CLIP}")

    shard_ids     = []
    policy_losses = []
    value_losses  = []
    total_losses  = []

    shard_pbar = tqdm(all_shards[start_shard:], desc="Overall shards", unit="shard")

    for shard_path in shard_pbar:
        shard_id = int(os.path.basename(shard_path).split("_")[1].split(".")[0])

        avg_p, avg_v, avg_t = train_shard(
            model, optimizer, scheduler, shard_path, shard_id
        )

        shard_ids.append(shard_id)
        policy_losses.append(avg_p)
        value_losses.append(avg_v)
        total_losses.append(avg_t)

        # step LR scheduler after each shard
        scheduler.step()

        # save latest checkpoint
        model.save_checkpoint(
            checkpoint_path, optimizer, epoch=EPOCHS, shard_id=shard_id
        )

        # save permanent per-shard checkpoint
        model.save_checkpoint(
            f"{OUTPUT_DIR}/shard_{shard_id:03d}.pt",
            optimizer, epoch=EPOCHS, shard_id=shard_id
        )

        # update loss plot
        save_loss_plot(shard_ids, policy_losses, value_losses, total_losses)

        shard_pbar.set_postfix(
            shard=shard_id,
            p=f"{avg_p:.3f}",
            v=f"{avg_v:.3f}"
        )

        torch.cuda.empty_cache()

    log.info("Training complete!")
    log.info(f"Final   — policy: {policy_losses[-1]:.4f}  value: {value_losses[-1]:.4f}")
    log.info(f"Best policy loss: {min(policy_losses):.4f} at shard {shard_ids[policy_losses.index(min(policy_losses))]}")
    log.info(f"Best value  loss: {min(value_losses):.4f}  at shard {shard_ids[value_losses.index(min(value_losses))]}")


if __name__ == "__main__":
    main()