import torch
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, optimizer, scheduler, loss_fn, device="cuda", 
                 gradient_clip=1.0, accumulation_steps=1):
        """
        Improved trainer with gradient accumulation and better monitoring
        
        Args:
            model: Chess model
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            loss_fn: Loss function
            device: Device to train on
            gradient_clip: Gradient clipping value
            accumulation_steps: Number of steps to accumulate gradients
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.gradient_clip = gradient_clip
        self.accumulation_steps = accumulation_steps
        
    def train_epoch(self, dataloader, epoch=None):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_acc = 0.0
        total_value_sign_acc = 0.0
        total_value_mae = 0.0
        num_batches = 0
        
        desc = f"Epoch {epoch}" if epoch is not None else "Training"
        loop = tqdm(dataloader, desc=desc, unit="batch")
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(loop):
            # Move data to device
            board = batch["board"].to(self.device)  # (B, 119, 8, 8)
            move = batch["move"].to(self.device)    # (B,)
            value = batch["value"].to(self.device)  # (B,)
            mask = batch["mask"].to(self.device)    # (B, 4672)
            
            # Forward pass
            policy_logits, value_pred = self.model(board)
            
            # Calculate loss
            loss, p_loss, v_loss, acc, sign_acc, mae = self.loss_fn(
                policy_logits, move, value_pred, value, mask
            )
            
            # Normalize loss for gradient accumulation
            loss = loss / self.accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights after accumulation steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.gradient_clip
                    )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item() * self.accumulation_steps
            total_policy += p_loss.item()
            total_value += v_loss.item()
            total_acc += acc.item()
            total_value_sign_acc += sign_acc.item()
            total_value_mae += mae.item()
            num_batches += 1
            
            # Update progress bar
            loop.set_postfix({
                "loss": f"{loss.item() * self.accumulation_steps:.3f}",
                "p": f"{p_loss.item():.3f}",
                "v": f"{v_loss.item():.3f}",
                "acc": f"{acc.item():.3f}",
                "sign": f"{sign_acc.item():.3f}",
            })
        
        # Step scheduler if provided
        if self.scheduler:
            self.scheduler.step()
        
        # Return average metrics
        return {
            "loss": total_loss / num_batches,
            "policy": total_policy / num_batches,
            "value": total_value / num_batches,
            "move_accuracy": total_acc / num_batches,
            "value_sign_accuracy": total_value_sign_acc / num_batches,
            "value_mae": total_value_mae / num_batches,
        }
    
    @torch.no_grad()
    def validate(self, dataloader):
        """Validation loop"""
        self.model.eval()
        
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_acc = 0.0
        total_value_sign_acc = 0.0
        total_value_mae = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Validation", unit="batch"):
            board = batch["board"].to(self.device)
            move = batch["move"].to(self.device)
            value = batch["value"].to(self.device)
            mask = batch["mask"].to(self.device)
            
            policy_logits, value_pred = self.model(board)
            
            loss, p_loss, v_loss, acc, sign_acc, mae = self.loss_fn(
                policy_logits, move, value_pred, value, mask
            )
            
            total_loss += loss.item()
            total_policy += p_loss.item()
            total_value += v_loss.item()
            total_acc += acc.item()
            total_value_sign_acc += sign_acc.item()
            total_value_mae += mae.item()
            num_batches += 1
        
        return {
            "loss": total_loss / num_batches,
            "policy": total_policy / num_batches,
            "value": total_value / num_batches,
            "move_accuracy": total_acc / num_batches,
            "value_sign_accuracy": total_value_sign_acc / num_batches,
            "value_mae": total_value_mae / num_batches,
        }