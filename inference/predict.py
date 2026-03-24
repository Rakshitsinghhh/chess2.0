import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import chess
import chess.pgn
from tqdm import tqdm
from model.chess_model import ChessModel
from utils.fen_utils import fen_to_tensor
from utils.generate_move_mask import generate_move_mask
from utils.move_index_encoding import policy_index_to_move, move_to_policy_index

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Core prediction (same as predict.py — no hardcoded rules)
# ─────────────────────────────────────────────────────────────────────────────

def predict(fen, model):
    board = chess.Board(fen)
    if board.is_checkmate():
        return None, (1.0 if board.turn == chess.BLACK else -1.0)
    if board.is_stalemate() or board.is_insufficient_material():
        return None, 0.0

    board_tensor = torch.tensor(fen_to_tensor(fen), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    mask         = torch.tensor(generate_move_mask(fen), dtype=torch.float32).to(DEVICE)

    if mask.sum().item() == 0:
        return list(board.legal_moves)[0].uci(), 0.0

    with torch.no_grad():
        policy_logits, value = model(board_tensor)

    policy_logits = policy_logits.squeeze(0) + (mask - 1) * 1e9
    best_idx      = torch.argmax(policy_logits).item()
    move_uci      = policy_index_to_move(best_idx).uci()

    legal_ucis = [m.uci() for m in board.legal_moves]
    if move_uci not in legal_ucis:
        logits_np  = policy_logits.cpu().numpy()
        best_score = -float('inf')
        move_uci   = legal_ucis[0]
        for lm in board.legal_moves:
            idx = move_to_policy_index(lm)
            if idx is not None and logits_np[idx] > best_score:
                best_score = logits_np[idx]
                move_uci   = lm.uci()

    # top-3 moves
    top3_indices = torch.topk(policy_logits, 3).indices.tolist()
    top3_moves   = []
    for idx in top3_indices:
        try:
            m = policy_index_to_move(idx).uci()
            if m in legal_ucis:
                top3_moves.append(m)
        except:
            pass

    score = float(torch.clamp(value, -1.0, 1.0).item())
    return move_uci, score, top3_moves


# ─────────────────────────────────────────────────────────────────────────────
# Test suite — structured categories with expected answers
# ─────────────────────────────────────────────────────────────────────────────

TEST_SUITE = [

    # ── Category 1: Opening ───────────────────────────────────────────────────
    {
        "category": "Opening",
        "name":     "Starting position",
        "fen":      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "best":     ["e2e4", "d2d4", "g1f3", "c2c4"],   # any of these = pass
        "score_min": -0.2, "score_max": 0.2,             # near zero = equal game
    },
    {
        "category": "Opening",
        "name":     "After 1.e4 — Black response",
        "fen":      "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "best":     ["e7e5", "c7c5", "e7e6", "c7c6"],
        "score_min": -0.2, "score_max": 0.2,
    },

    # ── Category 2: Checkmate detection ───────────────────────────────────────
    {
        "category": "Checkmate detection",
        "name":     "Already checkmate — no move",
        "fen":      "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
        "best":     [None],
        "score_min": 0.9, "score_max": 1.0,
    },
    {
        "category": "Checkmate detection",
        "name":     "Stalemate — no move",
        "fen":      "5bnr/4p1pq/4Qpkr/7p/7P/4P3/PPPP1PP1/RNB1KBNR b KQ - 2 10",
        "best":     [None],
        "score_min": -0.1, "score_max": 0.1,
    },

    # ── Category 3: Mate in 1 ─────────────────────────────────────────────────
    {
        "category": "Mate in 1",
        "name":     "Scholar's mate Qxf7",
        "fen":      "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "best":     ["h5f7"],
        "score_min": 0.9, "score_max": 1.0,
    },
    {
        "category": "Mate in 1",
        "name":     "King+Rook Kf7",
        "fen":      "7k/8/5K2/8/8/8/8/7R w - - 0 1",
        "best":     ["f6f7"],
        "score_min": 0.9, "score_max": 1.0,
    },
    {
        "category": "Mate in 1",
        "name":     "Queen smothered Qa8",
        "fen":      "7k/6pp/8/8/8/8/8/Q5K1 w - - 0 1",
        "best":     ["a1a8"],
        "score_min": 0.9, "score_max": 1.0,
    },
    {
        "category": "Mate in 1",
        "name":     "Two rooks Rc8",
        "fen":      "k7/8/1KR5/2R5/8/8/8/8 w - - 0 1",
        "best":     ["c6c8", "c5c8"],
        "score_min": 0.9, "score_max": 1.0,
    },
    {
        "category": "Mate in 1",
        "name":     "Rook smothered Ra8",
        "fen":      "7k/6pp/6p1/8/8/8/8/R5RK w - - 0 1",
        "best":     ["a1a8"],
        "score_min": 0.9, "score_max": 1.0,
    },

    # ── Category 4: Tactics ───────────────────────────────────────────────────
    {
        "category": "Tactics",
        "name":     "Knight fork Nd5",
        "fen":      "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 1",
        "best":     ["f3d5", "c3d5"],
        "score_min": 0.3, "score_max": 1.0,
    },
    {
        "category": "Tactics",
        "name":     "Bishop sac Bxf7",
        "fen":      "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 1",
        "best":     ["c4f7"],
        "score_min": 0.3, "score_max": 1.0,
    },
    {
        "category": "Tactics",
        "name":     "Discovered check Ng5",
        "fen":      "r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        "best":     ["f3g5"],
        "score_min": 0.3, "score_max": 1.0,
    },

    # ── Category 5: Endgame moves ─────────────────────────────────────────────
    {
        "category": "Endgame",
        "name":     "K+P vs K — Black king escorts pawn",
        "fen":      "8/8/8/8/8/3k4/3p4/3K4 b - - 0 1",
        "best":     ["d3c3", "d3e3", "d3c2", "d3e2"],   # any king move helping pawn
        "score_min": -1.0, "score_max": -0.3,            # Black winning
    },
    {
        "category": "Endgame",
        "name":     "Pawn promotion a8=Q",
        "fen":      "8/P7/8/8/8/8/8/4K1k1 w - - 0 1",
        "best":     ["a7a8q"],
        "score_min": 0.3, "score_max": 1.0,
    },
    {
        "category": "Endgame",
        "name":     "K+R vs K — rook cuts off king",
        "fen":      "8/8/8/8/8/3k4/8/3RK3 w - - 0 1",
        "best":     ["d1d3", "d1d4", "d1d5", "d1d6", "d1d7", "d1d8",
                     "d1a1", "d1b1", "d1c1", "d1e1", "d1f1", "d1g1", "d1h1"],
        "score_min": 0.3, "score_max": 1.0,
    },

    # ── Category 6: Stalemate avoidance ───────────────────────────────────────
    {
        "category": "Stalemate avoidance",
        "name":     "Don't play Qe8 stalemate",
        "fen":      "7k/8/6QK/8/8/8/8/8 w - - 0 1",
        "best":     ["!g6e8"],     # ! prefix = must NOT play this move
        "score_min": 0.5, "score_max": 1.0,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model):
    results = []

    for test in tqdm(TEST_SUITE, desc="Evaluating", unit="position"):
        fen      = test["fen"]
        expected = test["best"]
        s_min    = test["score_min"]
        s_max    = test["score_max"]

        result = predict(fen, model)
        if len(result) == 3:
            move, score, top3 = result
        else:
            move, score = result
            top3 = []

        # ── Check move correctness ────────────────────────────────────────────
        move_pass = False
        if expected == [None]:
            move_pass = (move is None)
        elif expected[0].startswith("!"):
            # "!" prefix = must NOT play this move
            forbidden = expected[0][1:]
            move_pass = (move != forbidden)
        else:
            # top-3 check: pass if correct move is in top 3
            move_pass = (move in expected) or any(m in expected for m in top3)

        # Strict check: is the exact best move correct
        strict_pass = False
        if expected == [None]:
            strict_pass = (move is None)
        elif expected[0].startswith("!"):
            strict_pass = (move != expected[0][1:])
        else:
            strict_pass = (move in expected)

        # ── Check score correctness ───────────────────────────────────────────
        score_pass = s_min <= score <= s_max

        results.append({
            "category":    test["category"],
            "name":        test["name"],
            "move":        move,
            "score":       score,
            "top3":        top3,
            "expected":    expected,
            "move_pass":   move_pass,
            "strict_pass": strict_pass,
            "score_pass":  score_pass,
            "pass":        strict_pass and score_pass,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results):
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    total_pass   = 0
    total_strict = 0
    total_score  = 0
    total        = len(results)

    for cat, items in categories.items():
        cat_pass   = sum(1 for r in items if r["pass"])
        cat_strict = sum(1 for r in items if r["strict_pass"])
        cat_score  = sum(1 for r in items if r["score_pass"])
        print(f"\n── {cat} ({cat_pass}/{len(items)} full pass) ──")

        for r in items:
            move_icon  = "✓" if r["strict_pass"] else ("~" if r["move_pass"] else "✗")
            score_icon = "✓" if r["score_pass"]  else "✗"
            expected_str = str(r["expected"]) if r["expected"] != [None] else "None"
            print(f"  [{move_icon}move {score_icon}score] {r['name']}")
            print(f"           played={r['move']}  score={r['score']:.3f}  expected={expected_str}")
            if r["top3"] and not r["strict_pass"] and r["move_pass"]:
                print(f"           top3={r['top3']} (correct move in top3)")

        total_pass   += cat_pass
        total_strict += cat_strict
        total_score  += cat_score

    print("\n" + "=" * 70)
    print(f"TOTAL:  {total_pass}/{total} full pass  |  "
          f"{total_strict}/{total} move correct  |  "
          f"{total_score}/{total} score correct")

    pct = total_pass / total * 100
    print(f"SCORE:  {pct:.1f}%")

    if pct >= 80:
        print("★ Excellent — model is playing strong chess")
    elif pct >= 60:
        print("◆ Good — solid play, some tactical gaps")
    elif pct >= 40:
        print("▲ Developing — needs more tactical training data")
    else:
        print("▼ Early stage — continue training")
    print("=" * 70)

    return total_pass, total


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, _, _, shard_id = ChessModel.load_checkpoint(
        "outputs/models/latest.pt", device=DEVICE
    )
    model.eval()
    print(f"Model loaded — shard {shard_id}\n")

    results = evaluate(model)
    passed, total = print_report(results)