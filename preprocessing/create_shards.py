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
# Config — tuned for low RAM
# ─────────────────────────────────────────────────────────────────────────────

GAME_CSV    = "data/processed/labeled_chess_moves.csv"
PUZZLE_CSV  = "data/processed/lichess_puzzle_transformed.csv"
OUTPUT_DIR  = "data/shards/"
SHARD_SIZE  = 10000   # samples per shard
MAX_PUZZLES = 500_000 # total unique puzzles to use
PUZZLE_REPS = 3       # repeat puzzle CSV this many times
CHUNK_SIZE  = 10_000  # rows per CSV chunk — small = low RAM


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

def make_sample(fen, move_uci, value_target):
    try:
        board_tensor = fen_to_tensor(fen)
        move_idx     = move_to_index(move_uci)
        move_mask    = generate_move_mask(fen)
    except Exception:
        return None
    return {
        "board": torch.tensor(board_tensor,        dtype=torch.float32),
        "move":  torch.tensor(move_idx,            dtype=torch.long),
        "mask":  torch.tensor(move_mask,           dtype=torch.float32),
        "value": torch.tensor(float(value_target), dtype=torch.float32),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shard writer
# ─────────────────────────────────────────────────────────────────────────────

class ShardWriter:
    """
    Holds exactly SHARD_SIZE samples in RAM at once.
    Flushes to disk immediately when full.
    Peak RAM = one shard = ~24MB.
    """
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
        random.shuffle(self.buffer)
        path = f"{self.output_dir}/shard_{self.shard_id:03d}.pt"
        torch.save(self.buffer, path)
        print(f"  shard_{self.shard_id:03d}.pt  ({len(self.buffer):,} samples)")
        self.buffer   = []
        self.shard_id += 1

    def close(self):
        self._flush()
        print(f"\n  Total shards : {self.shard_id}")
        print(f"  Total samples: {self.total:,}")


# ─────────────────────────────────────────────────────────────────────────────
# Stream game CSV directly to writer — zero RAM buildup
# ─────────────────────────────────────────────────────────────────────────────

def stream_games(writer, skipped):
    print(f"\n[Games] Streaming {GAME_CSV}...")
    total_rows = 0

    for chunk in pd.read_csv(GAME_CSV, chunksize=CHUNK_SIZE):
        for _, row in chunk.iterrows():
            try:
                fen  = row["fen"]
                move = row["engine_best"]
                try:    eval_cp = float(row["eval_before"])
                except: eval_cp = 0.0
                try:    winner  = str(row["winner"]).strip().lower()
                except: winner  = ""
                sample = make_sample(fen, move, blend_value(eval_cp, winner))
                if sample: writer.add(sample)
                else:      skipped[0] += 1
            except Exception:
                skipped[0] += 1

        total_rows += len(chunk)
        print(f"  [Games] {total_rows:,} rows...", end="\r")

    print(f"\n[Games] Done — {writer.total:,} samples")


# ─────────────────────────────────────────────────────────────────────────────
# Stream puzzle CSV directly to writer — zero RAM buildup
# Called PUZZLE_REPS times with different random seeds each pass
# ─────────────────────────────────────────────────────────────────────────────

def stream_puzzles_once(writer, skipped, pass_num):
    """
    Streams puzzle CSV once from disk — no list held in RAM.
    Each pass reads the CSV fresh with a different random skip
    so the order is different each time.
    RAM: only CHUNK_SIZE rows at a time (~few MB).
    """
    if not os.path.exists(PUZZLE_CSV):
        print(f"\n[Puzzles] NOT FOUND: {PUZZLE_CSV} — skipping")
        return

    total_loaded = 0
    total_rows   = 0

    for chunk in pd.read_csv(PUZZLE_CSV, chunksize=CHUNK_SIZE):
        rows = list(chunk.iterrows())
        random.shuffle(rows)  # shuffle within chunk each pass

        for _, row in rows:
            if total_loaded >= MAX_PUZZLES:
                break
            try:
                fen   = row["FEN"]
                moves = str(row["Moves"]).split()
                if len(moves) < 2:
                    skipped[0] += 1
                    continue

                board = chess.Board(fen)
                board.push(chess.Move.from_uci(moves[0]))
                puzzle_fen   = board.fen()
                correct_move = moves[1]

                # Cap at ±0.8 — prevents value head collapse
                value  = 0.8 if board.turn == chess.WHITE else -0.8
                sample = make_sample(puzzle_fen, correct_move, value)

                if sample:
                    writer.add(sample)
                    total_loaded += 1
                else:
                    skipped[0] += 1
            except Exception:
                skipped[0] += 1

        total_rows += len(chunk)
        print(f"  [Puzzles pass {pass_num}] {total_rows:,} rows, {total_loaded:,} encoded...", end="\r")

        if total_loaded >= MAX_PUZZLES:
            break

    print(f"\n[Puzzles pass {pass_num}] Done — {total_loaded:,} samples")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def create_shards():
    print("=" * 60)
    print("RAM-safe shard creation")
    print(f"  Game CSV    : {GAME_CSV}")
    print(f"  Puzzle CSV  : {PUZZLE_CSV}")
    print(f"  Output      : {OUTPUT_DIR}")
    print(f"  Shard size  : {SHARD_SIZE:,}")
    print(f"  Max puzzles : {MAX_PUZZLES:,} × {PUZZLE_REPS} passes")
    print(f"  Chunk size  : {CHUNK_SIZE:,} rows  (~few MB RAM)")
    print(f"  Peak RAM    : ~{SHARD_SIZE * 24 // 1024}MB (one shard buffer)")
    print("=" * 60)

    skipped = [0]
    writer  = ShardWriter(OUTPUT_DIR, SHARD_SIZE)

    # Pass 1: stream all game positions
    stream_games(writer, skipped)

    # Pass 2+: stream puzzle CSV PUZZLE_REPS times
    # Each pass re-reads from disk — no RAM held between passes
    for rep in range(1, PUZZLE_REPS + 1):
        print(f"\n[Puzzles] Pass {rep}/{PUZZLE_REPS}...")
        stream_puzzles_once(writer, skipped, rep)

    # Flush remaining samples
    writer.close()

    print(f"\n{'='*60}")
    print(f"Complete!")
    print(f"  Skipped : {skipped[0]:,}")
    print(f"\nNext steps:")
    print(f"  rm outputs/models/latest.pt")
    print(f"  python training/train.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    create_shards()