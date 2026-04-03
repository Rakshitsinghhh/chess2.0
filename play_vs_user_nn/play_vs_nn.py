import os
import sys
import time

import torch
import chess
import chess.pgn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.chess_model import ChessModel
from utils.fen_utils import fen_to_tensor
from utils.generate_move_mask import generate_move_mask
from utils.move_index_encoding import policy_index_to_move, move_to_policy_index

from bot.test_bot_connection import predict as nn_predict


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL_PATH = os.path.join(REPO_ROOT, "outputs/models/latest.pt")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def white_view_value(v_white: float) -> float:
    """Model value is from White's perspective."""
    return float(clamp(v_white, -1.0, 1.0))


def viewpoint_value(v_white: float, user_side: chess.Color) -> float:
    """Convert White-perspective value to the user's chosen viewpoint."""
    v_white = white_view_value(v_white)
    return v_white if user_side == chess.WHITE else -v_white


@torch.no_grad()
def nn_value_and_move_prob(fen: str, model: ChessModel, move_uci: str):
    """
    Single NN forward pass that returns:
      - value_before (White perspective, clamped to [-1, 1])
      - policy probability of `move_uci` (masked to legal moves)
    """
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        # If move is illegal, prob is 0 and value is still meaningful.
        illegal = True
    else:
        illegal = False

    board_tensor = torch.tensor(fen_to_tensor(fen), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    mask = torch.tensor(generate_move_mask(fen), dtype=torch.float32).to(DEVICE)

    policy_logits, v_before = model(board_tensor)
    policy_logits = policy_logits.squeeze(0)
    v_before = float(torch.clamp(v_before, -1.0, 1.0).item())

    masked_logits = policy_logits + (mask - 1) * 1e9
    probs = torch.softmax(masked_logits, dim=-1)

    if illegal:
        prob = 0.0
    else:
        idx = move_to_policy_index(move)
        prob = float(probs[idx].item())

    return v_before, prob


@torch.no_grad()
def nn_value_only(fen: str, model: ChessModel) -> float:
    """Single NN forward pass that returns value (White perspective, clamped)."""
    board_tensor = torch.tensor(fen_to_tensor(fen), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    _, v = model(board_tensor)
    return float(torch.clamp(v, -1.0, 1.0).item())


def load_model(model_path: str) -> ChessModel:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model, _, _, _ = ChessModel.load_checkpoint(model_path, device=DEVICE)
    model.eval()
    return model


def ask_user_side() -> chess.Color:
    while True:
        side = input("Choose your side: type `white` or `black`: ").strip().lower()
        if side in ("white", "w"):
            return chess.WHITE
        if side in ("black", "b"):
            return chess.BLACK
        print("Invalid input. Please type `white` or `black`.")


def ask_user_move(board: chess.Board, user_side: chess.Color) -> str:
    legal_ucis = {m.uci() for m in board.legal_moves}
    while True:
        s = input(f"Your move (UCI like e2e4, or 'quit'): ").strip().lower()
        if s in ("quit", "exit", "q"):
            return "quit"

        # Quick UX: detect if user typed SAN (like "Nf3" or "e4" without from-square).
        if len(s) in (2, 3) and s[0] in "abcdefgh":
            print("Tip: use full UCI like `e2e4` (from+to), not short algebraic like `e4`.")
            continue
        if any(ch in s for ch in ["x", "-", "+", "#"]) and (len(s) < 4 or not s[0].isdigit()):
            print("Tip: this game expects UCI (example: `g1f3`), not SAN (example: `Nf3`).")
            # Still fall through to UCI parser in case it's actually UCI.

        try:
            mv = chess.Move.from_uci(s)
        except Exception:
            print("Invalid UCI format.")
            continue

        if s in legal_ucis:
            return s

        # Provide a clearer reason for common mistakes.
        piece = board.piece_at(mv.from_square)
        if piece is None:
            print("Illegal move: no piece on the `from` square for that move.")
        elif piece.color != user_side:
            print(f"Illegal move: you're playing {('White' if user_side == chess.WHITE else 'Black')}, but that move changes the other side's piece.")
        else:
            print("Illegal move for this position. Try again.")

        # Show up to 10 legal moves to help the user quickly recover.
        sample = sorted(list(legal_ucis))[:10]
        print(f"Example legal moves: {', '.join(sample)}")


def main():
    model_path = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
    if not os.path.isabs(model_path):
        model_path = os.path.join(REPO_ROOT, model_path)
    model = load_model(model_path)

    user_side = ask_user_side()
    nn_side = chess.BLACK if user_side == chess.WHITE else chess.WHITE

    board = chess.Board()

    invalid_inputs = 0
    user_points = 0.0  # accumulate value deltas from user's viewpoint

    print("\nStarting match!")
    print(f"You play: {'White' if user_side == chess.WHITE else 'Black'}")
    print(f"NN plays: {'White' if nn_side == chess.WHITE else 'Black'}\n")

    game = chess.pgn.Game()
    node = game

    # Main loop
    while not board.is_game_over(claim_draw=True):
        print(board)
        print()

        if board.turn == user_side:
            user_uci = ask_user_move(board, user_side)
            if user_uci == "quit":
                print("You quit the game.")
                break
            move = chess.Move.from_uci(user_uci)

            # Rating: value delta from user's viewpoint.
            v_before_nn, prob_user = nn_value_and_move_prob(board.fen(), model, user_uci)
            v_before_user = viewpoint_value(v_before_nn, user_side)

            board.push(move)

            v_after_nn = nn_value_only(board.fen(), model)
            v_after_user = viewpoint_value(v_after_nn, user_side)
            delta = v_after_user - v_before_user

            # Points: scaled delta (small values feel better).
            user_points += clamp(delta * 100.0, -200.0, 200.0)

            node = node.add_variation(move)
            continue

        # NN turn
        fen_before = board.fen()
        start = time.time()
        nn_move_uci, _, nn_top_candidates = nn_predict(fen_before, model)
        think_ms = int((time.time() - start) * 1000)

        if nn_move_uci is None:
            print("NN returned no move (checkmate/stalemate).")
            break

        move = chess.Move.from_uci(nn_move_uci)
        if move not in board.legal_moves:
            # Safety fallback: should not happen if predict() is correct.
            fallback = next(iter(board.legal_moves))
            nn_move_uci = fallback.uci()
            move = fallback

        v_before_nn, prob = nn_value_and_move_prob(fen_before, model, nn_move_uci)
        v_before_user = viewpoint_value(v_before_nn, user_side)

        # Compute value after NN move without changing the board.
        btmp = board.copy()
        btmp.push(move)
        v_after_nn = nn_value_only(btmp.fen(), model)
        v_after_user = viewpoint_value(v_after_nn, user_side)

        print(f"NN move: {nn_move_uci}   (think {think_ms} ms)")
        print(f"  Value (before, your view): {v_before_user:+.3f}")
        print(f"  Value (after,  your view): {v_after_user:+.3f}")
        print(f"  Policy prob for move: {prob * 100:.2f}%")

        if nn_top_candidates:
            print(f"  NN top candidates: {', '.join(nn_top_candidates[:5])}")

        board.push(move)
        node = node.add_variation(move)

    # Final board
    print("\nFinal position:")
    print(board)

    result = board.outcome(claim_draw=True)
    if result is None:
        print("Game over.")
    else:
        print("Result:", result.termination)
        print("Board outcome:", result)

    print(f"\nRating (from your viewpoint): {user_points:+.0f} points")

    # Save PGN
    os.makedirs("outputs/pgn", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    pgn_path = os.path.join("outputs/pgn", f"nn_game_{ts}.pgn")
    with open(pgn_path, "w", encoding="utf-8") as f:
        f.write(str(game))
    print(f"Saved PGN: {pgn_path}")


if __name__ == "__main__":
    main()

