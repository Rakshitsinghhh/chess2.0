# training/endgame_trainer.py - Fixed version
"""Unified trainer that combines regular and endgame training into one model"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
import glob
import random
import gc
import numpy as np
from datetime import datetime

from model.chess_model import ChessModel
from training.loss import ChessLoss


class UnifiedDataset(Dataset):
    """Dataset that combines regular and endgame positions"""
    
    def __init__(self, regular_shard_paths, endgame_data, config):
        self.regular_shard_paths = regular_shard_paths
        self.endgame_data = endgame_data
        self.config = config
        self.batch_ratio = config.get('batch_ratio', 0.3)
        self.regular_data_cache = []
        self.current_shard_idx = 0
        self._load_next_regular_batch()
        
    def _load_next_regular_batch(self):
        if self.current_shard_idx < len(self.regular_shard_paths):
            try:
                data = torch.load(self.regular_shard_paths[self.current_shard_idx], 
                                weights_only=False, map_location='cpu')
                max_per_shard = 3000
                if len(data) > max_per_shard:
                    data = random.sample(data, max_per_shard)
                
                # Ensure all data is converted to tensors with correct types
                converted_data = []
                for sample in data:
                    board = sample["board"]
                    move = sample["move"]
                    mask = sample["mask"]
                    value = sample["value"]
                    
                    # Convert to tensors
                    if isinstance(board, np.ndarray):
                        board = torch.from_numpy(board).float()
                    elif isinstance(board, torch.Tensor):
                        board = board.float()
                    else:
                        board = torch.tensor(board, dtype=torch.float32)
                    
                    if isinstance(move, np.ndarray):
                        move = torch.from_numpy(move).long()
                    elif isinstance(move, torch.Tensor):
                        move = move.long()
                    else:
                        move = torch.tensor(move, dtype=torch.long)
                    
                    if isinstance(mask, np.ndarray):
                        mask = torch.from_numpy(mask).float()
                    elif isinstance(mask, torch.Tensor):
                        mask = mask.float()
                    else:
                        mask = torch.tensor(mask, dtype=torch.float32)
                    
                    if isinstance(value, np.ndarray):
                        value = torch.from_numpy(value).float()
                    elif isinstance(value, torch.Tensor):
                        value = value.float()
                    else:
                        value = torch.tensor(value, dtype=torch.float32)
                    
                    converted_data.append({
                        "board": board,
                        "move": move,
                        "mask": mask,
                        "value": value
                    })
                
                self.regular_data_cache = converted_data
                self.current_shard_idx += 1
                return True
            except Exception as e:
                print(f"Failed to load: {e}")
                self.regular_data_cache = []
        return False
    
    def __len__(self):
        return min(len(self.regular_shard_paths) * 3000, 50000) + len(self.endgame_data)
    
    def __getitem__(self, idx):
        if not self.regular_data_cache and self.current_shard_idx < len(self.regular_shard_paths):
            self._load_next_regular_batch()
        
        use_endgame = (self.endgame_data and random.random() < self.batch_ratio)
        
        if use_endgame and self.endgame_data:
            sample = random.choice(self.endgame_data)
        elif self.regular_data_cache:
            sample = random.choice(self.regular_data_cache)
        else:
            # Return dummy sample with correct tensor types
            return (torch.zeros(20, 8, 8, dtype=torch.float32), 
                   torch.tensor(0, dtype=torch.long),
                   torch.zeros(4672, dtype=torch.float32),
                   torch.tensor(0.0, dtype=torch.float32))
        
        # Ensure all are tensors with correct types
        board = sample["board"]
        move = sample["move"]
        mask = sample["mask"]
        value = sample["value"]
        
        # Convert to tensors if they're numpy arrays
        if isinstance(board, np.ndarray):
            board = torch.from_numpy(board).float()
        elif not isinstance(board, torch.Tensor):
            board = torch.tensor(board, dtype=torch.float32)
        else:
            board = board.float()
        
        if isinstance(move, np.ndarray):
            move = torch.from_numpy(move).long()
        elif not isinstance(move, torch.Tensor):
            move = torch.tensor(move, dtype=torch.long)
        else:
            move = move.long()
        
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
        elif not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)
        else:
            mask = mask.float()
        
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value).float()
        elif not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32)
        else:
            value = value.float()
        
        return board, move, mask, value


class UnifiedTrainer:
    """Unified trainer that continues training on the same model"""
    
    def __init__(self, model, config, checkpoint_path="outputs/models/latest.pt"):
        self.model = model
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = next(model.parameters()).device
        self.logger = logging.getLogger(__name__)
        self.endgame_data = []
        self._load_endgame_data()
        
    def _load_endgame_data(self):
        """Load endgame positions from shards"""
        endgame_shards = sorted(glob.glob(f"{self.config['endgame_shard_dir']}/endgame_shard_*.pt"))
        
        if not endgame_shards:
            self.logger.warning(f"No endgame shards found in {self.config['endgame_shard_dir']}")
            return
        
        self.logger.info(f"Loading endgame shards...")
        total_loaded = 0
        max_endgame = 10000  # Limit to 10,000 for memory
        
        for shard_path in endgame_shards:
            try:
                data = torch.load(shard_path, weights_only=False, map_location='cpu')
                self.logger.info(f"Loaded {len(data)} positions from {os.path.basename(shard_path)}")
                
                for sample in data:
                    # Convert to tensors with correct types
                    board = sample["board"]
                    move = sample["move"]
                    mask = sample["mask"]
                    value = sample["value"]
                    
                    if isinstance(board, np.ndarray):
                        board = torch.from_numpy(board).float()
                    elif isinstance(board, torch.Tensor):
                        board = board.float()
                    else:
                        board = torch.tensor(board, dtype=torch.float32)
                    
                    if isinstance(move, np.ndarray):
                        move = torch.from_numpy(move).long()
                    elif isinstance(move, torch.Tensor):
                        move = move.long()
                    else:
                        move = torch.tensor(move, dtype=torch.long)
                    
                    if isinstance(mask, np.ndarray):
                        mask = torch.from_numpy(mask).float()
                    elif isinstance(mask, torch.Tensor):
                        mask = mask.float()
                    else:
                        mask = torch.tensor(mask, dtype=torch.float32)
                    
                    if isinstance(value, np.ndarray):
                        value = torch.from_numpy(value).float()
                    elif isinstance(value, torch.Tensor):
                        value = value.float()
                    else:
                        value = torch.tensor(value, dtype=torch.float32)
                    
                    self.endgame_data.append({
                        "board": board,
                        "move": move,
                        "mask": mask,
                        "value": value
                    })
                    total_loaded += 1
                    
                    if total_loaded >= max_endgame:
                        break
                
                if total_loaded >= max_endgame:
                    self.logger.info(f"Reached {max_endgame} endgame positions")
                    break
                    
            except Exception as e:
                self.logger.warning(f"Failed to load {shard_path}: {e}")
        
        self.logger.info(f"Total endgame positions loaded: {len(self.endgame_data)}")
    
    def train_epoch(self, regular_shard_paths, epoch_num, optimizer):
        """Train one unified epoch"""
        
        if len(self.endgame_data) == 0:
            self.logger.warning("No endgame data, skipping epoch")
            return {"move_accuracy": 0.0, "total_loss": 0.0}
        
        dataset = UnifiedDataset(regular_shard_paths, self.endgame_data, self.config)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
        
        loss_fn = ChessLoss(
            value_weight=self.config.get('value_weight', 0.05),
            label_smoothing=self.config.get('label_smoothing', 0.1)
        )
        
        self.model.train()
        total_acc = 0.0
        total_loss = 0.0
        n = 0
        
        pbar = tqdm(loader, desc=f"Unified Epoch {epoch_num}", unit="batch")
        optimizer.zero_grad()
        
        accumulation_steps = self.config.get('accumulation_steps', 4)
        
        for batch_idx, batch_data in enumerate(pbar):
            # Unpack batch
            boards, moves, masks, values = batch_data
            
            # Move to device
            boards = boards.to(self.device)
            moves = moves.to(self.device)
            masks = masks.to(self.device)
            values = values.to(self.device)
            
            # Forward pass
            policy_logits, value_pred = self.model(boards)
            
            # Calculate loss
            loss, p_loss, v_loss, acc, sign_acc, mae = loss_fn(
                policy_logits, moves, value_pred, values, masks
            )
            
            # Backward pass
            (loss / accumulation_steps).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_acc += acc.item()
            total_loss += loss.item()
            n += 1
            
            pbar.set_postfix(acc=f"{acc.item():.3f}", loss=f"{loss.item():.3f}")
            
            if batch_idx % 200 == 0:
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
        
        return {
            "move_accuracy": total_acc / n if n > 0 else 0.0,
            "total_loss": total_loss / n if n > 0 else 0.0,
        }
    
    def run_unified_training(self, regular_shard_paths, start_epoch=0):
        """Run unified training to combine regular + endgame knowledge"""
        
        if len(self.endgame_data) == 0:
            self.logger.warning("No endgame data available!")
            return []
        
        self.logger.info("="*60)
        self.logger.info("UNIFIED TRAINING - COMBINING REGULAR + ENDGAME")
        self.logger.info("="*60)
        self.logger.info(f"Endgame positions: {len(self.endgame_data)}")
        self.logger.info(f"Regular shards: {len(regular_shard_paths)}")
        self.logger.info(f"Batch ratio (endgame): {self.config.get('batch_ratio', 0.3):.0%}")
        self.logger.info(f"Epochs: {self.config.get('epochs', 5)}")
        self.logger.info(f"Saving to: {self.checkpoint_path}")
        self.logger.info("="*60)
        
        # Setup optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        
        all_metrics = []
        epochs = self.config.get('epochs', 5)
        
        for epoch in range(start_epoch, epochs):
            self.logger.info(f"\n{'='*40}")
            self.logger.info(f"Unified Epoch {epoch}")
            self.logger.info(f"{'='*40}")
            
            metrics = self.train_epoch(regular_shard_paths, epoch, optimizer)
            all_metrics.append(metrics)
            
            self.logger.info(f"✓ Epoch {epoch} | Acc: {metrics['move_accuracy']:.3f} | Loss: {metrics['total_loss']:.3f}")
            
            # Save to latest.pt (updates the unified model)
            torch.save({
                "model_state": self.model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "unified_epoch": epoch,
                "unified_metrics": metrics,
                "last_updated": datetime.now().isoformat(),
                "is_unified_model": True
            }, self.checkpoint_path)
            self.logger.info(f"✓ Unified model saved to {self.checkpoint_path}")
            
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        self.logger.info("="*60)
        self.logger.info("UNIFIED TRAINING COMPLETE!")
        self.logger.info(f"Model saved to: {self.checkpoint_path}")
        self.logger.info("="*60)
        
        return all_metrics