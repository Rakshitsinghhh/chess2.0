import chess
import numpy as np
from utils.move_index_encoding import move_to_policy_index

TOTAL_MOVES = 4672  # AlphaZero-style move space


def generate_move_mask(fen):
    """
    Input: FEN string
    Output: binary mask of size (4672,)
            1 → legal move
            0 → illegal move
    """
    board = chess.Board(fen)
    mask = np.zeros(TOTAL_MOVES, dtype=np.float32)

    for move in board.legal_moves:
        try:
            idx = move_to_policy_index(move)
            mask[idx] = 1
        except:
            # Skip moves not in encoding (edge cases)
            continue

    return mask