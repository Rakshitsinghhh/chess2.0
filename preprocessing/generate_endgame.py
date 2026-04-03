# preprocessing/generate_endgame.py - SIMPLE RELIABLE VERSION
import chess
import chess.engine
import torch
import os
import random
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import time
import subprocess

class SimpleReliableEndgameGenerator:
    """Simple, reliable endgame generator with proper error handling"""
    
    def __init__(self, output_dir="data/endgame_shards"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Check Stockfish
        self.stockfish_path = self._find_stockfish()
        self.use_stockfish = self.stockfish_path is not None
        
        # Statistics
        self.stats = {
            'total_attempts': 0,
            'valid_positions': 0,
            'invalid': 0,
            'stockfish_errors': 0
        }
    
    def _find_stockfish(self):
        """Find Stockfish executable"""
        import shutil
        paths = ["/usr/games/stockfish", "/usr/local/bin/stockfish", "stockfish"]
        for path in paths:
            if shutil.which(path) or os.path.exists(path):
                return shutil.which(path) or path
        return None
    
    def get_best_move_simple(self, fen, depth=16):
        """Get best move with proper engine lifecycle management"""
        if not self.use_stockfish:
            return None, 0.0
        
        try:
            # Create new engine for each position (safer)
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path, timeout=5) as engine:
                board = chess.Board(fen)
                
                # Skip game over positions
                if board.is_game_over():
                    return None, 0.0
                
                # Analyze position
                try:
                    analysis = engine.analyse(
                        board, 
                        chess.engine.Limit(depth=depth),
                        multipv=1
                    )
                    
                    if not analysis or not analysis[0]['pv']:
                        return None, 0.0
                    
                    best_move = analysis[0]['pv'][0]
                    score = analysis[0]['score']
                    
                    # Get evaluation
                    if score.relative.is_mate():
                        eval_value = 1.0 if score.relative.mate() > 0 else -1.0
                    else:
                        cp = score.relative.score()
                        eval_value = max(-1.0, min(1.0, cp / 500.0))
                    
                    return best_move, eval_value
                    
                except Exception as e:
                    self.stats['stockfish_errors'] += 1
                    return None, 0.0
                    
        except Exception as e:
            self.stats['stockfish_errors'] += 1
            return None, 0.0
    
    def _safe_square(self, file_idx, rank_idx):
        """Create safe square with bounds"""
        file_idx = max(0, min(7, file_idx))
        rank_idx = max(0, min(7, rank_idx))
        return chess.square(file_idx, rank_idx)
    
    def generate_pawn_endgames(self, target=2000):
        """Generate pawn endgames"""
        positions = []
        
        scenarios = [
            'passed_pawn', 'king_in_front', 'opposition', 'pawn_race'
        ]
        
        print(f"\nGenerating pawn endgames...")
        
        with tqdm(total=target, desc="Pawn endgames") as pbar:
            while len(positions) < target:
                self.stats['total_attempts'] += 1
                
                board = chess.Board()
                board.clear()
                scenario = random.choice(scenarios)
                
                try:
                    if scenario == 'passed_pawn':
                        # Create a passed pawn
                        pawn_file = random.randint(0, 7)
                        pawn_rank = random.randint(4, 5)
                        pawn_square = self._safe_square(pawn_file, pawn_rank)
                        board.set_piece_at(pawn_square, chess.Piece(chess.PAWN, chess.WHITE))
                        
                        # White king supporting
                        wk_file = max(0, min(7, pawn_file + random.choice([-1, 0, 1])))
                        wk_rank = pawn_rank - 1
                        wk_square = self._safe_square(wk_file, wk_rank)
                        board.set_piece_at(wk_square, chess.Piece(chess.KING, chess.WHITE))
                        
                        # Black king blocking
                        bk_file = max(0, min(7, pawn_file + random.choice([-1, 0, 1])))
                        bk_rank = pawn_rank + 1
                        if bk_rank <= 7:
                            bk_square = self._safe_square(bk_file, bk_rank)
                            board.set_piece_at(bk_square, chess.Piece(chess.KING, chess.BLACK))
                    
                    elif scenario == 'opposition':
                        # King opposition
                        wk_file = random.randint(2, 5)
                        wk_rank = random.randint(3, 4)
                        wk_square = self._safe_square(wk_file, wk_rank)
                        board.set_piece_at(wk_square, chess.Piece(chess.KING, chess.WHITE))
                        
                        bk_rank = wk_rank + 2 if wk_rank < 5 else wk_rank - 2
                        bk_square = self._safe_square(wk_file, bk_rank)
                        board.set_piece_at(bk_square, chess.Piece(chess.KING, chess.BLACK))
                        
                        # Add pawn
                        pawn_rank = wk_rank + 1
                        if pawn_rank <= 6:
                            pawn_square = self._safe_square(wk_file, pawn_rank)
                            board.set_piece_at(pawn_square, chess.Piece(chess.PAWN, chess.WHITE))
                    
                    else:
                        # Simple pawn endgame
                        wk_square = self._safe_square(random.randint(2, 5), 1)
                        board.set_piece_at(wk_square, chess.Piece(chess.KING, chess.WHITE))
                        
                        bk_square = self._safe_square(random.randint(2, 5), 6)
                        board.set_piece_at(bk_square, chess.Piece(chess.KING, chess.BLACK))
                        
                        pawn_square = self._safe_square(random.randint(0, 7), random.randint(4, 5))
                        board.set_piece_at(pawn_square, chess.Piece(chess.PAWN, chess.WHITE))
                    
                    fen = board.fen()
                    
                    # Quick validation
                    if board.is_valid() and not board.is_game_over():
                        # Get best move
                        best_move, eval_value = self.get_best_move_simple(fen)
                        
                        if best_move and best_move in board.legal_moves:
                            positions.append({
                                'fen': fen,
                                'type': 'pawn',
                                'scenario': scenario,
                                'best_move': best_move.uci(),
                                'eval': eval_value
                            })
                            self.stats['valid_positions'] += 1
                            pbar.update(1)
                        else:
                            self.stats['invalid'] += 1
                    else:
                        self.stats['invalid'] += 1
                        
                except Exception as e:
                    self.stats['invalid'] += 1
                    continue
        
        return positions
    
    def generate_rook_endgames(self, target=1500):
        """Generate rook endgames"""
        positions = []
        
        scenarios = ['standard', 'corner', 'cutoff']
        
        print(f"\nGenerating rook endgames...")
        
        with tqdm(total=target, desc="Rook endgames") as pbar:
            while len(positions) < target:
                self.stats['total_attempts'] += 1
                
                board = chess.Board()
                board.clear()
                scenario = random.choice(scenarios)
                
                try:
                    # White king
                    wk_square = self._safe_square(random.randint(2, 5), 1)
                    board.set_piece_at(wk_square, chess.Piece(chess.KING, chess.WHITE))
                    
                    # Black king (corner)
                    corners = [self._safe_square(0, 7), self._safe_square(7, 7),
                              self._safe_square(0, 0), self._safe_square(7, 0)]
                    bk_square = random.choice(corners)
                    board.set_piece_at(bk_square, chess.Piece(chess.KING, chess.BLACK))
                    
                    # White rook
                    rook_square = self._safe_square(random.randint(0, 7), random.randint(3, 4))
                    board.set_piece_at(rook_square, chess.Piece(chess.ROOK, chess.WHITE))
                    
                    fen = board.fen()
                    
                    if board.is_valid() and not board.is_game_over():
                        best_move, eval_value = self.get_best_move_simple(fen)
                        
                        if best_move and best_move in board.legal_moves:
                            positions.append({
                                'fen': fen,
                                'type': 'rook',
                                'scenario': scenario,
                                'best_move': best_move.uci(),
                                'eval': eval_value
                            })
                            self.stats['valid_positions'] += 1
                            pbar.update(1)
                        else:
                            self.stats['invalid'] += 1
                    else:
                        self.stats['invalid'] += 1
                        
                except Exception as e:
                    self.stats['invalid'] += 1
                    continue
        
        return positions
    
    def generate_queen_endgames(self, target=1000):
        """Generate queen endgames"""
        positions = []
        
        scenarios = ['standard', 'corner_mate']
        
        print(f"\nGenerating queen endgames...")
        
        with tqdm(total=target, desc="Queen endgames") as pbar:
            while len(positions) < target:
                self.stats['total_attempts'] += 1
                
                board = chess.Board()
                board.clear()
                scenario = random.choice(scenarios)
                
                try:
                    # White king
                    wk_square = self._safe_square(random.randint(2, 5), 1)
                    board.set_piece_at(wk_square, chess.Piece(chess.KING, chess.WHITE))
                    
                    # Black king (corner for mate practice)
                    corners = [self._safe_square(0, 7), self._safe_square(7, 7)]
                    bk_square = random.choice(corners)
                    board.set_piece_at(bk_square, chess.Piece(chess.KING, chess.BLACK))
                    
                    # White queen (not giving immediate check)
                    queen_square = self._safe_square(random.randint(3, 4), random.randint(3, 4))
                    board.set_piece_at(queen_square, chess.Piece(chess.QUEEN, chess.WHITE))
                    
                    fen = board.fen()
                    
                    # Skip stalemate
                    if chess.Board(fen).is_stalemate():
                        self.stats['invalid'] += 1
                        continue
                    
                    if board.is_valid() and not board.is_game_over():
                        best_move, eval_value = self.get_best_move_simple(fen)
                        
                        if best_move and best_move in board.legal_moves:
                            positions.append({
                                'fen': fen,
                                'type': 'queen',
                                'scenario': scenario,
                                'best_move': best_move.uci(),
                                'eval': eval_value
                            })
                            self.stats['valid_positions'] += 1
                            pbar.update(1)
                        else:
                            self.stats['invalid'] += 1
                    else:
                        self.stats['invalid'] += 1
                        
                except Exception as e:
                    self.stats['invalid'] += 1
                    continue
        
        return positions
    
    def create_shards(self, all_positions, shard_size=1000):
        """Create shards"""
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        
        try:
            from utils.fen_utils import fen_to_tensor
            from utils.generate_move_mask import generate_move_mask
            from utils.move_index_encoding import move_to_policy_index
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Make sure you're running from the project root directory")
            return
        
        if not all_positions:
            print("No positions to create shards!")
            return
        
        num_shards = (len(all_positions) + shard_size - 1) // shard_size
        
        print(f"\nCreating {num_shards} shards...")
        
        for shard_idx in range(num_shards):
            start = shard_idx * shard_size
            end = min((shard_idx + 1) * shard_size, len(all_positions))
            
            shard_data = []
            errors = 0
            
            for pos in tqdm(all_positions[start:end], desc=f"Shard {shard_idx+1}/{num_shards}"):
                try:
                    fen = pos['fen']
                    best_move = chess.Move.from_uci(pos['best_move'])
                    
                    # Verify
                    board = chess.Board(fen)
                    if best_move not in board.legal_moves:
                        errors += 1
                        continue
                    
                    move_idx = move_to_policy_index(best_move)
                    if move_idx is None:
                        errors += 1
                        continue
                    
                    sample = {
                        "board": fen_to_tensor(fen),
                        "move": move_idx,
                        "mask": generate_move_mask(fen),
                        "value": float(pos['eval'])
                    }
                    shard_data.append(sample)
                    
                except Exception as e:
                    errors += 1
                    continue
            
            if shard_data:
                shard_path = os.path.join(self.output_dir, f"endgame_shard_{shard_idx:04d}.pt")
                torch.save(shard_data, shard_path)
                print(f"✅ Saved {len(shard_data)} positions to {shard_path}")
                if errors > 0:
                    print(f"   Skipped {errors} invalid positions")
    
    def print_stats(self):
        """Print statistics"""
        print("\n" + "="*60)
        print("GENERATION STATISTICS")
        print("="*60)
        print(f"Total attempts: {self.stats['total_attempts']}")
        print(f"Valid positions saved: {self.stats['valid_positions']}")
        print(f"Invalid positions skipped: {self.stats['invalid']}")
        print(f"Stockfish errors: {self.stats['stockfish_errors']}")
        
        if self.stats['total_attempts'] > 0:
            success_rate = (self.stats['valid_positions'] / self.stats['total_attempts']) * 100
            print(f"\nSuccess rate: {success_rate:.1f}%")
        print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("SIMPLE RELIABLE ENDGAME DATASET GENERATOR")
    print("="*60)
    
    # Check Stockfish
    import shutil
    stockfish_path = shutil.which("stockfish")
    
    if not stockfish_path:
        print("\n⚠️ Stockfish not found! Installing...")
        os.system("sudo apt-get update && sudo apt-get install -y stockfish")
        stockfish_path = shutil.which("stockfish")
    
    if stockfish_path:
        print(f"✅ Stockfish found at: {stockfish_path}")
    else:
        print("❌ Stockfish not available. Cannot generate reliable endgame data.")
        exit(1)
    
    generator = SimpleReliableEndgameGenerator()
    
    try:
        # Generate smaller set first to test
        print("\n🚀 Generating endgame positions...")
        
        all_positions = []
        
        # Start with smaller numbers to test
        all_positions.extend(generator.generate_pawn_endgames(1000))
        all_positions.extend(generator.generate_rook_endgames(500))
        all_positions.extend(generator.generate_queen_endgames(500))
        
        print(f"\n✅ Generated {len(all_positions)} verified endgame positions")
        
        if all_positions:
            # Shuffle
            random.shuffle(all_positions)
            
            # Create shards
            generator.create_shards(all_positions, shard_size=1000)
            
            # Print statistics
            generator.print_stats()
            
            # Save metadata
            metadata = {
                'total_positions': len(all_positions),
                'generated_at': datetime.now().isoformat(),
                'stats': generator.stats
            }
            with open(os.path.join(generator.output_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print("\n" + "="*60)
            print("✅ ENDGAME DATASET GENERATION COMPLETE!")
            print(f"   Location: {generator.output_dir}")
            print(f"   Total positions: {len(all_positions)}")
            print("="*60)
            
            # Show sample
            if all_positions:
                print("\n📋 Sample positions:")
                for i, pos in enumerate(all_positions[:3]):
                    print(f"{i+1}. {pos['type']} - {pos['scenario']}")
                    print(f"   FEN: {pos['fen'][:50]}...")
                    print(f"   Best move: {pos['best_move']}")
                    print()
        else:
            print("\n❌ No positions generated. Check Stockfish installation.")
            print("Try running: stockfish --version")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()