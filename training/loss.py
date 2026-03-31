import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessLoss(nn.Module):
    def __init__(self, value_weight=0.05, label_smoothing=0.1, use_focal_loss=False, focal_gamma=2.0):
        """
        Improved loss function for chess training
        
        Args:
            value_weight: Weight for value loss vs policy loss
            label_smoothing: Label smoothing for policy loss (helps with generalization)
            use_focal_loss: Use focal loss to focus on hard examples
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.value_weight = value_weight
        self.label_smoothing = label_smoothing
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        
    def forward(self, policy_logits, target_moves, value_pred, target_values, move_mask):
        """
        Args:
            policy_logits: (B, 4672) raw logits from policy head
            target_moves: (B,) ground truth move index
            value_pred: (B, 1) predicted position value
            target_values: (B,) ground truth value in [-1, +1]
            move_mask: (B, 4672) 1 = legal, 0 = illegal
        """
        # Mask illegal moves (set to very negative value)
        masked_logits = policy_logits.masked_fill(move_mask == 0, -1e9)
        
        # Policy loss with label smoothing
        if self.label_smoothing > 0:
            # Convert targets to one-hot with smoothing
            n_classes = policy_logits.size(-1)
            with torch.no_grad():
                smooth_targets = torch.zeros_like(policy_logits).scatter_(
                    1, target_moves.unsqueeze(1), 1.0
                )
                smooth_targets = smooth_targets * (1 - self.label_smoothing) + \
                                self.label_smoothing / n_classes
                # Mask illegal moves in targets too
                smooth_targets = smooth_targets * move_mask
                # Renormalize
                smooth_targets = smooth_targets / smooth_targets.sum(dim=-1, keepdim=True)
            
            policy_loss = -(smooth_targets * F.log_softmax(masked_logits, dim=-1)).sum(dim=-1).mean()
        else:
            policy_loss = F.cross_entropy(masked_logits, target_moves)
        
        # Optional: Focal loss for hard examples
        if self.use_focal_loss and not self.label_smoothing:
            ce_loss = F.cross_entropy(masked_logits, target_moves, reduction='none')
            pt = torch.exp(-ce_loss)
            policy_loss = ((1 - pt) ** self.focal_gamma * ce_loss).mean()
        
        # Value loss with Huber loss for robustness
        value_pred = value_pred.squeeze(-1)  # (B, 1) -> (B,)
        # Use Huber loss instead of MSE for outliers
        value_loss = F.huber_loss(value_pred, target_values, delta=0.5)
        
        # Total loss
        total_loss = policy_loss + self.value_weight * value_loss
        
        # Track accuracy for monitoring
        with torch.no_grad():
            # Get predicted move (after masking)
            pred_moves = torch.argmax(masked_logits, dim=-1)
            move_accuracy = (pred_moves == target_moves).float().mean()
            
            # Value sign accuracy
            value_sign_acc = ((value_pred * target_values) > 0).float().mean()
            
            # Value magnitude error
            value_mae = torch.abs(value_pred - target_values).mean()
        
        return total_loss, policy_loss, value_loss, move_accuracy, value_sign_acc, value_mae