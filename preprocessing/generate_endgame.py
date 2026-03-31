# preprocessing/generate_endgame.py - Fixed square generation
import chess
import chess.engine
import torch
import os
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from functools import partial
import json

class ProfessionalEndgameGenerator:
    """Professional endgame dataset generator with optimal move selection"""
    
    def __init__(self, output_dir="data/endgame_shards", use_stockfish=True):
        self.output_dir = output_dir
        self.use_stockfish = use_stockfish
        os.makedirs(output_dir, exist_ok=True)
        
        # Stockfish path (try common locations)
        self.stockfish_path = self._find_stockfish()
        
        # Cache for evaluations
        self.eval_cache = {}
        
    def _find_stockfish(self):
        """Find Stockfish executable"""
        possible_paths = [
            "/usr/games/stockfish",
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish",
            "stockfish"
        ]
        
        for path in possible_paths:
            if os.path.exists(path) or os.system(f"which {path} > /dev/null 2>&1") == 0:
                return path
        return None
    
    def _safe_square(self, file_idx, rank_idx):
        """Safely create a square with bounds checking"""
        file_idx = max(0, min(7, file_idx))
        rank_idx = max(0, min(7, rank_idx))
        return chess.square(file_idx, rank_idx)
    
    def generate_pawn_endgames(self, num_positions=5000):
        """Generate comprehensive K+P vs K positions"""
        positions = []
        
        print("Generating pawn endgames...")
        for _ in tqdm(range(num_positions), desc="Pawn endgames"):
            board = chess.Board()
            board.clear()
            
            # Strategic placement
            scenario = random.choice([
                'normal', 'blocked', 'passed_pawn', 'king_in_front', 'distant_pawn'
            ])
            
            try:
                if scenario == 'normal':
                    wk_file = random.randint(2, 5)  # Center files
                    wk_rank = random.randint(1, 2)
                    wk_square = self._safe_square(wk_file, wk_rank)
                    board.set_piece_at(wk_square, chess.Piece(chess.KING, chess.WHITE))
                    
                    bk_file = random.randint(2, 5)
                    bk_rank = random.randint(6, 7)
                    bk_square = self._safe_square(bk_file, bk_rank)
                    board.set_piece_at(bk_square, chess.Piece(chess.KING, chess.BLACK))
                    
                    pawn_file = random.randint(0, 7)
                    pawn_rank = random.randint(4, 6)
                    pawn_square = self._safe_square(pawn_file, pawn_rank)
                    board.set_piece_at(pawn_square, chess.Piece(chess.PAWN, chess.WHITE))
                    
                elif scenario == 'passed_pawn':
                    # Create passed pawn with no enemy pawns in front
                    pawn_file = random.randint(0, 7)
                    pawn_rank = random.randint(4, 6)
                    pawn_square = self._safe_square(pawn_file, pawn_rank)
                    board.set_piece_at(pawn_square, chess.Piece(chess.PAWN, chess.WHITE))
                    
                    wk_file = pawn_file + random.choice([-1, 0, 1])
                    wk_file = max(0, min(7, wk_file))
                    wk_rank = pawn_rank - random.randint(1, 2)
                    wk_rank = max(1, min(2, wk_rank))
                    wk_square = self._safe_square(wk_file, wk_rank)
                    board.set_piece_at(wk_square, chess.Piece(chess.KING, chess.WHITE))
                    
                    bk_file = pawn_file + random.choice([-2, -1, 0, 1, 2])
                    bk_file = max(0, min(7, bk_file))
                    bk_rank = pawn_rank + random.randint(1, 2)
                    bk_rank = min(7, bk_rank)
                    bk_square = self._safe_square(bk_file, bk_rank)
                    board.set_piece_at(bk_square, chess.Piece(chess.KING, chess.BLACK))
                    
                elif scenario == 'king_in_front':
                    # King directly in front of pawn
                    pawn_file = random.randint(0, 7)
                    pawn_rank = random.randint(3, 5)
                    pawn_square = self._safe_square(pawn_file, pawn_rank)
                    board.set_piece_at(pawn_square, chess.Piece(chess.PAWN, chess.WHITE))
                    
                    wk_file = pawn_file
                    wk_rank = pawn_rank + 1
                    wk_rank = min(7, wk_rank)
                    wk_square = self._safe_square(wk_file, wk_rank)
                    board.set_piece_at(wk_square, chess.Piece(chess.KING, chess.WHITE))
                    
                    bk_file = pawn_file + random.choice([-2, -1, 0, 1, 2])
                    bk_file = max(0, min(7, bk_file))
                    bk_rank = pawn_rank + random.randint(2, 3)
                    bk_rank = min(7, bk_rank)
                    bk_square = self._safe_square(bk_file, bk_rank)
                    board.set_piece_at(bk_square, chess.Piece(chess.KING, chess.BLACK))
                
                # Only add if position is legal
                if board.is_valid() and not board.is_checkmate() and not board.is_stalemate():
                    positions.append({
                        'fen': board.fen(),
                        'type': 'pawn',
                        'scenario': scenario
                    })
            except Exception as e:
                continue  # Skip invalid positions
        
        return positions
    
    def generate_rook_endgames(self, num_positions=5000):
        """Generate comprehensive K+R vs K positions"""
        positions = []
        
        print("Generating rook endgames...")
        for _ in tqdm(range(num_positions), desc="Rook endgames"):
            board = chess.Board()
            board.clear()
            
            try:
                scenario = random.choice([
                    'cutoff', 'ladder', 'corner', 'edge', 'central'
                ])
                
                # White king placement
                wk_file = random.randint(2, 5)
                wk_rank = random.randint(1, 2)
                wk_square = self._safe_square(wk_file, wk_rank)
                board.set_piece_at(wk_square, chess.Piece(chess.KING, chess.WHITE))
                
                # Black king placement
                if scenario == 'corner':
                    bk_corners = [self._safe_square(0, 7), self._safe_square(7, 7), 
                                 self._safe_square(0, 0), self._safe_square(7, 0)]
                    bk_square = random.choice(bk_corners)
                elif scenario == 'edge':
                    bk_file = random.choice([0, 7])
                    bk_rank = random.randint(1, 6)
                    bk_square = self._safe_square(bk_file, bk_rank)
                else:
                    bk_file = random.randint(0, 7)
                    bk_rank = random.randint(5, 7)
                    bk_square = self._safe_square(bk_file, bk_rank)
                board.set_piece_at(bk_square, chess.Piece(chess.KING, chess.BLACK))
                
                # Rook placement (not giving immediate check)
                rook_file = random.randint(0, 7)
                rook_rank = random.randint(3, 6)
                rook_square = self._safe_square(rook_file, rook_rank)
                
                # Ensure rook doesn't give check
                board.set_piece_at(rook_square, chess.Piece(chess.ROOK, chess.WHITE))
                if board.is_check():
                    # Move rook to safe square
                    safe_squares = [self._safe_square(1, 1), self._safe_square(2, 2), 
                                   self._safe_square(6, 6), self._safe_square(5, 5)]
                    for square in safe_squares:
                        board.remove_piece_at(rook_square)
                        board.set_piece_at(square, chess.Piece(chess.ROOK, chess.WHITE))
                        if not board.is_check():
                            break
                
                if board.is_valid() and not board.is_checkmate():
                    positions.append({
                        'fen': board.fen(),
                        'type': 'rook',
                        'scenario': scenario
                    })
            except Exception as e:
                continue
        
        return positions
    
    def generate_queen_endgames(self, num_positions=5000):
        """Generate comprehensive K+Q vs K positions with stalemate avoidance"""
        positions = []
        
        print("Generating queen endgames...")
        for _ in tqdm(range(num_positions), desc="Queen endgames"):
            board = chess.Board()
            board.clear()
            
            try:
                # Critical stalemate scenarios
                scenario = random.choice([
                    'normal', 'near_stalemate', 'corner', 'edge', 'distance'
                ])
                
                # White king
                wk_file = random.randint(2, 5)
                wk_rank = random.randint(1, 2)
                wk_square = self._safe_square(wk_file, wk_rank)
                board.set_piece_at(wk_square, chess.Piece(chess.KING, chess.WHITE))
                
                # Black king placement (often in corner for stalemate risk)
                if scenario == 'near_stalemate':
                    # Place black king in corner with limited squares
                    bk_corners = [self._safe_square(0, 7), self._safe_square(7, 7), 
                                 self._safe_square(0, 0), self._safe_square(7, 0)]
                    bk_square = random.choice(bk_corners)
                elif scenario == 'corner':
                    bk_corners = [self._safe_square(0, 7), self._safe_square(7, 7)]
                    bk_square = random.choice(bk_corners)
                else:
                    bk_file = random.randint(0, 7)
                    bk_rank = random.randint(5, 7)
                    bk_square = self._safe_square(bk_file, bk_rank)
                board.set_piece_at(bk_square, chess.Piece(chess.KING, chess.BLACK))
                
                # Queen placement (avoid immediate check/stalemate)
                queen_squares = [self._safe_square(3, 3), self._safe_square(4, 3), 
                               self._safe_square(3, 4), self._safe_square(4, 4)]
                queen_square = random.choice(queen_squares)
                board.set_piece_at(queen_square, chess.Piece(chess.QUEEN, chess.WHITE))
                
                # Check for stalemate
                if board.is_stalemate():
                    # Adjust queen position
                    adjustments = [(1,0), (-1,0), (0,1), (0,-1)]
                    for df, dr in adjustments:
                        new_file = queen_square // 8 + df
                        new_rank = queen_square % 8 + dr
                        if 0 <= new_file <= 7 and 0 <= new_rank <= 7:
                            new_square = self._safe_square(new_file, new_rank)
                            board.remove_piece_at(queen_square)
                            board.set_piece_at(new_square, chess.Piece(chess.QUEEN, chess.WHITE))
                            if not board.is_stalemate():
                                break
                
                if board.is_valid() and not board.is_checkmate() and not board.is_stalemate():
                    positions.append({
                        'fen': board.fen(),
                        'type': 'queen',
                        'scenario': scenario
                    })
            except Exception as e:
                continue
        
        return positions
    
    def generate_bishop_endgames(self, num_positions=3000):
        """Generate K+B vs K and K+BB vs K positions"""
        positions = []
        
        print("Generating bishop endgames...")
        for _ in tqdm(range(num_positions), desc="Bishop endgames"):
            board = chess.Board()
            board.clear()
            
            try:
                num_bishops = random.choice([1, 2])
                same_color = random.choice([True, False]) if num_bishops == 2 else False
                
                # White king
                wk_square = self._safe_square(random.randint(3, 5), 1)
                board.set_piece_at(wk_square, chess.Piece(chess.KING, chess.WHITE))
                
                # Black king (corner)
                bk_corners = [self._safe_square(0, 7), self._safe_square(7, 7), 
                             self._safe_square(0, 0), self._safe_square(7, 0)]
                bk_square = random.choice(bk_corners)
                board.set_piece_at(bk_square, chess.Piece(chess.KING, chess.BLACK))
                
                # Place bishops
                for i in range(num_bishops):
                    if same_color:
                        # Same color bishops (light squares)
                        light_squares = [self._safe_square(2, 2), self._safe_square(4, 2), 
                                       self._safe_square(6, 2)]
                        bishop_square = random.choice(light_squares)
                    else:
                        # Opposite colors
                        if i == 0:
                            bishop_square = self._safe_square(2, 2)  # Light square
                        else:
                            bishop_square = self._safe_square(3, 3)  # Dark square
                    board.set_piece_at(bishop_square, chess.Piece(chess.BISHOP, chess.WHITE))
                
                if board.is_valid() and not board.is_checkmate():
                    positions.append({
                        'fen': board.fen(),
                        'type': 'bishop',
                        'num_bishops': num_bishops,
                        'same_color': same_color
                    })
            except Exception as e:
                continue
        
        return positions
    
    def generate_knight_endgames(self, num_positions=3000):
        """Generate K+N vs K and K+NN vs K positions"""
        positions = []
        
        print("Generating knight endgames...")
        for _ in tqdm(range(num_positions), desc="Knight endgames"):
            board = chess.Board()
            board.clear()
            
            try:
                num_knights = random.choice([1, 2])
                
                # White king
                wk_square = self._safe_square(random.randint(3, 5), 1)
                board.set_piece_at(wk_square, chess.Piece(chess.KING, chess.WHITE))
                
                # Black king
                bk_file = random.randint(0, 7)
                bk_rank = random.randint(5, 7)
                bk_square = self._safe_square(bk_file, bk_rank)
                board.set_piece_at(bk_square, chess.Piece(chess.KING, chess.BLACK))
                
                # Place knights
                knight_squares = [self._safe_square(2, 3), self._safe_square(3, 3), 
                                 self._safe_square(4, 3), self._safe_square(5, 3)]
                for i in range(num_knights):
                    knight_square = random.choice(knight_squares)
                    board.set_piece_at(knight_square, chess.Piece(chess.KNIGHT, chess.WHITE))
                
                if board.is_valid() and not board.is_checkmate():
                    positions.append({
                        'fen': board.fen(),
                        'type': 'knight',
                        'num_knights': num_knights
                    })
            except Exception as e:
                continue
        
        return positions
    
    def generate_complex_endgames(self, num_positions=4000):
        """Generate complex endgames with multiple pieces"""
        positions = []
        
        print("Generating complex endgames...")
        for _ in tqdm(range(num_positions), desc="Complex endgames"):
            board = chess.Board()
            board.clear()
            
            try:
                # Random piece configuration
                white_pieces = []
                
                # White pieces (material advantage but not overwhelming)
                piece_counts = {
                    chess.PAWN: random.randint(0, 2),
                    chess.KNIGHT: random.randint(0, 1),
                    chess.BISHOP: random.randint(0, 1),
                    chess.ROOK: random.randint(0, 1),
                    chess.QUEEN: random.choice([0, 0, 1])  # Queen rarely
                }
                
                # Place white pieces
                for piece_type, count in piece_counts.items():
                    for _ in range(count):
                        file_idx = random.randint(0, 7)
                        rank_idx = random.randint(1, 6)
                        square = self._safe_square(file_idx, rank_idx)
                        if not board.piece_at(square):
                            board.set_piece_at(square, chess.Piece(piece_type, chess.WHITE))
                            white_pieces.append((square, piece_type))
                
                # Black king (always present)
                bk_file = random.randint(0, 7)
                bk_rank = random.randint(5, 7)
                bk_square = self._safe_square(bk_file, bk_rank)
                board.set_piece_at(bk_square, chess.Piece(chess.KING, chess.BLACK))
                
                # White king
                wk_square = self._safe_square(random.randint(3, 5), 1)
                board.set_piece_at(wk_square, chess.Piece(chess.KING, chess.WHITE))
                
                # Ensure position is not trivially winning
                if board.is_valid() and not board.is_checkmate() and len(white_pieces) <= 4:
                    positions.append({
                        'fen': board.fen(),
                        'type': 'complex',
                        'white_pieces': len(white_pieces),
                        'piece_composition': str([p[1] for p in white_pieces])
                    })
            except Exception as e:
                continue
        
        return positions
    
    def get_best_move_with_stockfish(self, board, time_limit=0.2):
        """Get best move using Stockfish with proper error handling"""
        if not self.use_stockfish or not self.stockfish_path:
            return self._get_heuristic_best_move(board)
        
        try:
            # Cache check
            fen = board.fen()
            if fen in self.eval_cache:
                return self.eval_cache[fen]
            
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path, timeout=2) as engine:
                # Get analysis
                analysis = engine.analyse(board, chess.engine.Limit(time=time_limit), multipv=1)
                best_move = analysis['pv'][0]
                
                # Cache result
                self.eval_cache[fen] = best_move
                return best_move
        except Exception as e:
            # Fallback to heuristics
            return self._get_heuristic_best_move(board)
    
    def _get_heuristic_best_move(self, board):
        """Advanced heuristic move selection for endgames"""
        best_move = None
        best_score = -float('inf')
        
        for move in board.legal_moves:
            score = 0
            
            # Promotion (highest priority)
            if move.promotion:
                promotion_bonus = {
                    chess.QUEEN: 20,
                    chess.ROOK: 10,
                    chess.BISHOP: 5,
                    chess.KNIGHT: 5
                }
                score += promotion_bonus.get(move.promotion, 10)
            
            # Capture analysis
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                if captured:
                    capture_value = {
                        chess.QUEEN: 9,
                        chess.ROOK: 5,
                        chess.BISHOP: 3,
                        chess.KNIGHT: 3,
                        chess.PAWN: 1,
                        chess.KING: 0
                    }
                    score += capture_value.get(captured.piece_type, 0) * 2
            
            # Check/Checkmate
            board_copy = board.copy()
            board_copy.push(move)
            if board_copy.is_checkmate():
                score += 1000
            elif board_copy.is_check():
                score += 5
            
            # King safety and activity
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.KING:
                # Move king towards center or enemy king
                enemy_king = board.king(not board.turn)
                if enemy_king:
                    old_dist = chess.square_distance(move.from_square, enemy_king)
                    new_dist = chess.square_distance(move.to_square, enemy_king)
                    score += (old_dist - new_dist) * 3
                
                # Avoid moving into check
                if board_copy.is_check():
                    score -= 10
            
            # Piece centralization
            if piece and piece.piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                center_dist = min(
                    chess.square_distance(move.to_square, self._safe_square(3, 3)),
                    chess.square_distance(move.to_square, self._safe_square(3, 4)),
                    chess.square_distance(move.to_square, self._safe_square(4, 3)),
                    chess.square_distance(move.to_square, self._safe_square(4, 4))
                )
                score += (7 - center_dist)
            
            # Pawn advancement
            if piece and piece.piece_type == chess.PAWN:
                # Forward movement bonus
                if piece.color == chess.WHITE:
                    score += (chess.square_rank(move.to_square) - chess.square_rank(move.from_square)) * 2
                else:
                    score += (chess.square_rank(move.from_square) - chess.square_rank(move.to_square)) * 2
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def evaluate_position(self, board):
        """Evaluate position for value head training"""
        if self.use_stockfish and self.stockfish_path:
            try:
                with chess.engine.SimpleEngine.popen_uci(self.stockfish_path, timeout=2) as engine:
                    analysis = engine.analyse(board, chess.engine.Limit(time=0.1))
                    score = analysis['score']
                    
                    if score.relative.is_mate():
                        eval_value = 1.0 if score.relative.mate() > 0 else -1.0
                    else:
                        cp = score.relative.score()
                        # Convert to [-1, 1] range with sigmoid
                        eval_value = np.tanh(cp / 500.0)
                    
                    # Adjust for turn
                    if board.turn == chess.BLACK:
                        eval_value = -eval_value
                    
                    return eval_value
            except:
                pass
        
        # Fallback to material evaluation
        return self._material_evaluation(board)
    
    def _material_evaluation(self, board):
        """Simple material-based evaluation"""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }
        
        white_score = 0
        black_score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                value = piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    white_score += value
                else:
                    black_score += value
        
        total = white_score - black_score
        # Normalize to [-1, 1]
        eval_value = max(-1.0, min(1.0, total / 10.0))
        
        if board.turn == chess.BLACK:
            eval_value = -eval_value
        
        return eval_value
    
    def create_training_sample(self, position_info):
        """Create a complete training sample from position"""
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from utils.fen_utils import fen_to_tensor
        from utils.generate_move_mask import generate_move_mask
        from utils.move_index_encoding import move_to_policy_index
        
        fen = position_info['fen']
        board = chess.Board(fen)
        
        # Get best move using Stockfish or heuristics
        best_move = self.get_best_move_with_stockfish(board)
        
        if best_move:
            move_idx = move_to_policy_index(best_move)
            if move_idx is not None:
                sample = {
                    "board": fen_to_tensor(fen),
                    "move": move_idx,
                    "mask": generate_move_mask(fen),
                    "value": self.evaluate_position(board)
                }
                return sample
        return None
    
    def create_shards(self, all_positions, shard_size=2000):
        """Create optimized shards with balanced position types"""
        
        print(f"\nCreating {len(all_positions) // shard_size + 1} shards...")
        
        # Group by type for balanced shards
        positions_by_type = defaultdict(list)
        for pos in all_positions:
            positions_by_type[pos['type']].append(pos)
        
        # Create balanced shards
        num_shards = (len(all_positions) + shard_size - 1) // shard_size
        
        for shard_idx in range(num_shards):
            shard_data = []
            
            # Sample evenly from each type
            for pos_type, positions in positions_by_type.items():
                start = shard_idx * (shard_size // len(positions_by_type))
                end = start + (shard_size // len(positions_by_type))
                sampled = positions[start:end]
                
                for pos in tqdm(sampled, desc=f"Shard {shard_idx+1}/{num_shards} - {pos_type}"):
                    sample = self.create_training_sample(pos)
                    if sample:
                        shard_data.append(sample)
            
            # Save shard
            if shard_data:
                shard_path = os.path.join(self.output_dir, f"endgame_shard_{shard_idx:04d}.pt")
                torch.save(shard_data, shard_path)
                print(f"✅ Saved {len(shard_data)} positions to {shard_path}")
    
    def save_metadata(self, all_positions):
        """Save metadata about generated dataset"""
        metadata = {
            'total_positions': len(all_positions),
            'position_types': dict(),
            'scenarios': defaultdict(int),
            'generation_params': {
                'use_stockfish': self.use_stockfish,
                'stockfish_path': self.stockfish_path
            }
        }
        
        for pos in all_positions:
            pos_type = pos.get('type', 'unknown')
            metadata['position_types'][pos_type] = metadata['position_types'].get(pos_type, 0) + 1
            
            scenario = pos.get('scenario', 'normal')
            metadata['scenarios'][scenario] += 1
        
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total positions: {metadata['total_positions']}")
        print(f"   Position types: {dict(metadata['position_types'])}")
        print(f"   Scenarios: {dict(metadata['scenarios'])}")

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("PROFESSIONAL ENDGAME DATASET GENERATOR")
    print("="*60)
    
    # Initialize generator
    generator = ProfessionalEndgameGenerator(use_stockfish=False)  # Set to False initially to avoid Stockfish issues
    
    # Generate comprehensive endgame dataset
    print("\n🚀 Generating comprehensive endgame dataset...")
    print("   This will take 10-15 minutes for 30,000+ positions\n")
    
    all_positions = []
    
    # Generate all endgame types
    all_positions.extend(generator.generate_pawn_endgames(1000))  # Start smaller for testing
    all_positions.extend(generator.generate_rook_endgames(1000))
    all_positions.extend(generator.generate_queen_endgames(1000))
    all_positions.extend(generator.generate_bishop_endgames(500))
    all_positions.extend(generator.generate_knight_endgames(500))
    all_positions.extend(generator.generate_complex_endgames(1000))
    
    print(f"\n✅ Generated {len(all_positions)} total positions")
    
    # Shuffle for better distribution
    random.shuffle(all_positions)
    
    # Save metadata
    generator.save_metadata(all_positions)
    
    # Create optimized shards
    generator.create_shards(all_positions, shard_size=1000)
    
    print("\n" + "="*60)
    print("✅ ENDGAME DATASET COMPLETE!")
    print(f"   Location: {generator.output_dir}")
    print("   Ready for training with --endgame-weight 2.0")
    print("="*60)