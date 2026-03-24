import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessLoss(nn.Module):
    def __init__(self, value_weight=0.05):
        """
        value_weight: how much to weight value loss vs policy loss.
                      0.05 keeps policy dominant early in training
                      when cross-entropy is ~7.0 and MSE is ~0.5.
                      Tune between 0.01 (policy-heavy) and 0.1 (more value).
        """
        super().__init__()
        self.mse          = nn.MSELoss()
        self.value_weight = value_weight

    def forward(self, policy_logits, target_moves, value_pred, target_values, move_mask):
        """
        policy_logits  : (B, 4672)  — raw logits from policy head
        target_moves   : (B,)       — ground truth move index
        value_pred     : (B, 1)     — predicted position value
        target_values  : (B,)       — ground truth value in [-1, +1]
        move_mask      : (B, 4672)  — 1 = legal, 0 = illegal
        """

        # ── Mask illegal moves before computing policy loss ───────────────────
        policy_logits = policy_logits.masked_fill(move_mask == 0, -1e9)

        # ── Policy loss — cross entropy over legal moves ──────────────────────
        policy_loss = F.cross_entropy(policy_logits, target_moves)

        # ── Value loss — MSE between predicted and target value ───────────────
        value_pred  = value_pred.squeeze(-1)          # (B,1) → (B,)
        value_loss  = self.mse(value_pred, target_values)

        # ── Total loss — policy dominant, value secondary ─────────────────────
        total_loss = policy_loss + self.value_weight * value_loss

        return total_loss, policy_loss, value_loss