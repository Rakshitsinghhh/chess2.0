import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer, loss_fn, device="cuda"):
        self.model     = model.to(device)
        self.optimizer = optimizer
        self.loss_fn   = loss_fn
        self.device    = device

    def train_epoch(self, dataloader):
        self.model.train()

        total_loss   = 0.0
        total_policy = 0.0
        total_value  = 0.0

        loop = tqdm(dataloader)

        for batch in loop:
            board = batch["board"].to(self.device)   # (B, 20, 8, 8)
            move  = batch["move"].to(self.device)    # (B,)
            value = batch["value"].to(self.device)   # (B,)
            mask  = batch["mask"].to(self.device)    # (B, 4672)

            # ── Forward ───────────────────────────────────────────────────────
            policy_logits, value_pred = self.model(board)

            # ── Loss ──────────────────────────────────────────────────────────
            loss, p_loss, v_loss = self.loss_fn(
                policy_logits, move, value_pred, value, mask
            )

            # ── Backprop ──────────────────────────────────────────────────────
            self.optimizer.zero_grad()
            loss.backward()

            # gradient clipping — prevents exploding gradients in deep residual net
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss   += loss.item()
            total_policy += p_loss.item()
            total_value  += v_loss.item()

            loop.set_postfix({
                "loss":   f"{loss.item():.4f}",
                "policy": f"{p_loss.item():.4f}",
                "value":  f"{v_loss.item():.4f}",
            })

        n = len(dataloader)
        return {
            "loss":   total_loss   / n,
            "policy": total_policy / n,
            "value":  total_value  / n,
        }