import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import torch
import math
import chess
import random
from tqdm import tqdm

from utils.fen_utils import fen_to_tensor
from utils.move_index_encoding import move_to_index
from utils.generate_move_mask import generate_move_mask


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

GAME_CSV    = "data/processed/labeled_chess_moves.csv"
PUZZLE_CSV  = "data/processed/lichess_puzzle_transformed.csv"
OUTPUT_DIR  = "data/shards/"
SHARD_SIZE  = 10000       # samples per shard file
MAX_PUZZLES = 500_000     # cap puzzle rows to avoid RAM overload
CHUNK_SIZE  = 50_000      # rows read from CSV at a time — controls RAM usage


# ─────────────────────────────────────────────────────────────────────────────
# Value helpers
# ─────────────────────────────────────────────────────────────────────────────

def eval_to_value(eval_cp):
    try:
        return math.tanh(float(eval_cp) / 400.0)
    except:
        return 0.0

def winner_to_value(winner):
    if isinstance(winner, str):
        w = winner.strip().lower()
        if w == "white": return  1.0
        if w == "black": return -1.0
    return 0.0

def blend_value(eval_cp, winner):
    return 0.7 * eval_to_value(eval_cp) + 0.3 * winner_to_value(winner)


# ─────────────────────────────────────────────────────────────────────────────
# Encode one position → sample dict
# ─────────────────────────────────────────────────────────────────────────────

def make_sample(fen, move_uci, value_target):
    """
    Applies all 4 encodings:
      1. fen_to_tensor    → (20,8,8) board tensor
      2. move_to_index    → int policy index 0–4671
      3. generate_move_mask → (4672,) legal move mask
      4. value_target     → float in [-1, +1]
    Returns None if move is invalid.
    """
    try:
        board_tensor = fen_to_tensor(fen)
        move_idx     = move_to_index(move_uci)
        move_mask    = generate_move_mask(fen)
    except Exception:
        return None

    return {
        "board": torch.tensor(board_tensor,        dtype=torch.float32),  # (20,8,8)
        "move":  torch.tensor(move_idx,            dtype=torch.long),     # scalar
        "mask":  torch.tensor(move_mask,           dtype=torch.float32),  # (4672,)
        "value": torch.tensor(float(value_target), dtype=torch.float32),  # scalar
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shard writer — streams directly to disk, never holds all data in RAM
# ─────────────────────────────────────────────────────────────────────────────

class ShardWriter:
    def __init__(self, output_dir, shard_size):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.shard_id   = 0
        self.buffer     = []
        self.total      = 0
        os.makedirs(output_dir, exist_ok=True)

    def add(self, sample):
        self.buffer.append(sample)
        self.total += 1
        if len(self.buffer) >= self.shard_size:
            self._flush()

    def _flush(self):
        if not self.buffer:
            return
        path = f"{self.output_dir}/shard_{self.shard_id:03d}.pt"
        torch.save(self.buffer, path)
        print(f"  Saved shard_{self.shard_id:03d}.pt  ({len(self.buffer):,} samples)")
        self.buffer   = []
        self.shard_id += 1

    def close(self):
        self._flush()  # save any remaining samples
        print(f"\n  Total shards  : {self.shard_id}")
        print(f"  Total samples : {self.total:,}")


# ─────────────────────────────────────────────────────────────────────────────
# Source 1 — Game positions (chunked to save RAM)
# ─────────────────────────────────────────────────────────────────────────────

def process_game_csv(writer, skipped):
    """
    Reads labeled_chess_moves.csv in chunks of CHUNK_SIZE rows.
    Encodes each position and streams directly into ShardWriter.
    Never loads the full CSV into RAM.
    """
    print(f"\n[Games] Reading {GAME_CSV} in chunks of {CHUNK_SIZE:,}...")

    total_rows = 0
    for chunk in pd.read_csv(GAME_CSV, chunksize=CHUNK_SIZE):
        for _, row in chunk.iterrows():
            try:
                fen  = row["fen"]
                move = row["engine_best"]

                try:
                    eval_cp = float(row["eval_before"])
                except (KeyError, ValueError, TypeError):
                    eval_cp = 0.0

                try:
                    winner = str(row["winner"]).strip().lower()
                except (KeyError, TypeError):
                    winner = ""

                value  = blend_value(eval_cp, winner)
                sample = make_sample(fen, move, value)

                if sample:
                    writer.add(sample)
                else:
                    skipped[0] += 1

            except Exception:
                skipped[0] += 1

        total_rows += len(chunk)
        print(f"  [Games] Processed {total_rows:,} rows...", end="\r")

    print(f"\n[Games] Done — {total_rows:,} rows processed")


# ─────────────────────────────────────────────────────────────────────────────
# Source 2 — Puzzle positions (chunked to save RAM)
# ─────────────────────────────────────────────────────────────────────────────

def process_puzzle_csv(writer, skipped):
    """
    Reads lichess_puzzle_transformed.csv in chunks of CHUNK_SIZE rows.

    Puzzle format:
        FEN   = position before opponent's move
        Moves = space-separated UCI
                moves[0] = opponent move  → apply to get puzzle position
                moves[1] = correct answer → policy target

    Value target:
        +1.0 if White to move (White is making the winning move)
        -1.0 if Black to move (Black is making the winning move)
    """
    if not os.path.exists(PUZZLE_CSV):
        print(f"\n[Puzzles] NOT FOUND: {PUZZLE_CSV} — skipping puzzles")
        return

    print(f"\n[Puzzles] Reading {PUZZLE_CSV} in chunks of {CHUNK_SIZE:,}...")

    total_rows   = 0
    total_loaded = 0

    for chunk in pd.read_csv(PUZZLE_CSV, chunksize=CHUNK_SIZE):
        for _, row in chunk.iterrows():
            if total_loaded >= MAX_PUZZLES:
                break
            try:
                fen   = row["FEN"]
                moves = str(row["Moves"]).split()

                if len(moves) < 2:
                    skipped[0] += 1
                    continue

                # Apply opponent's move → reach actual puzzle position
                board = chess.Board(fen)
                board.push(chess.Move.from_uci(moves[0]))
                puzzle_fen   = board.fen()
                correct_move = moves[1]

                # Side to move is making the winning move
                value = 1.0 if board.turn == chess.WHITE else -1.0

                sample = make_sample(puzzle_fen, correct_move, value)

                if sample:
                    writer.add(sample)
                    total_loaded += 1
                else:
                    skipped[0] += 1

            except Exception:
                skipped[0] += 1

        total_rows += len(chunk)
        print(f"  [Puzzles] Processed {total_rows:,} rows, encoded {total_loaded:,}...", end="\r")

        if total_loaded >= MAX_PUZZLES:
            break

    print(f"\n[Puzzles] Done — {total_loaded:,} puzzles encoded")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def create_shards():
    print("=" * 60)
    print("Creating shards from 2 sources")
    print(f"  Game CSV    : {GAME_CSV}")
    print(f"  Puzzle CSV  : {PUZZLE_CSV}")
    print(f"  Output dir  : {OUTPUT_DIR}")
    print(f"  Shard size  : {SHARD_SIZE:,} samples")
    print(f"  Chunk size  : {CHUNK_SIZE:,} rows (RAM control)")
    print(f"  Max puzzles : {MAX_PUZZLES:,}")
    print("=" * 60)

    skipped = [0]
    writer  = ShardWriter(OUTPUT_DIR, SHARD_SIZE)

    # Stream game positions directly to shards
    process_game_csv(writer, skipped)

    # Stream puzzle positions directly to shards
    process_puzzle_csv(writer, skipped)

    # Flush any remaining samples
    writer.close()

    print(f"  Skipped     : {skipped[0]:,}")
    print("\nAll shards saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    create_shards()