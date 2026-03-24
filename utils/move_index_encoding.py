import chess

# Directions (dx, dy)
DIRS = [
    (0, 1), (1, 0), (0, -1), (-1, 0),   # rook-like
    (1, 1), (1, -1), (-1, -1), (-1, 1)  # bishop-like
]

KNIGHT_DIRS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2)
]

PROMOTION_PIECES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]


def square_to_coords(sq):
    return chess.square_file(sq), chess.square_rank(sq)


def coords_to_square(x, y):
    if 0 <= x < 8 and 0 <= y < 8:
        return chess.square(x, y)
    else:
        raise ValueError("Out of board")


# -------------------------
# MAIN WRAPPER (IMPORTANT FIX)
# -------------------------

def move_to_index(move):
    """
    Accepts:
        - string "e2e4"
        - chess.Move object

    Returns:
        index (0–4671)
    """
    if isinstance(move, str):
        move = chess.Move.from_uci(move)

    return move_to_policy_index(move)


# -------------------------
# ENCODER
# -------------------------

def move_to_policy_index(move: chess.Move) -> int:
    from_sq = move.from_square
    to_sq = move.to_square

    fx, fy = square_to_coords(from_sq)
    tx, ty = square_to_coords(to_sq)

    dx = tx - fx
    dy = ty - fy

    base = from_sq * 73

    # --- Promotion ---
    if move.promotion is not None:
        if move.promotion not in PROMOTION_PIECES:
            raise ValueError("Unsupported promotion piece")

        direction = dx + 1   # -1,0,1 → 0,1,2
        promo_idx = PROMOTION_PIECES.index(move.promotion)

        return base + 64 + direction * 4 + promo_idx

    # --- Knight ---
    if (dx, dy) in KNIGHT_DIRS:
        return base + 56 + KNIGHT_DIRS.index((dx, dy))

    # --- Sliding ---
    for d_idx, (ddx, ddy) in enumerate(DIRS):
        for dist in range(1, 8):
            if dx == ddx * dist and dy == ddy * dist:
                return base + d_idx * 7 + (dist - 1)

    raise ValueError(f"Illegal move for encoding: {move}")


# -------------------------
# DECODER
# -------------------------

def policy_index_to_move(index: int) -> chess.Move:
    from_sq = index // 73
    offset = index % 73

    fx, fy = square_to_coords(from_sq)

    # --- Sliding ---
    if offset < 56:
        d = offset // 7
        dist = (offset % 7) + 1
        dx, dy = DIRS[d]

        tx, ty = fx + dx * dist, fy + dy * dist
        return chess.Move(from_sq, coords_to_square(tx, ty))

    # --- Knight ---
    if offset < 64:
        dx, dy = KNIGHT_DIRS[offset - 56]
        tx, ty = fx + dx, fy + dy
        return chess.Move(from_sq, coords_to_square(tx, ty))

    # --- Promotion ---
    promo = offset - 64
    direction = promo // 4
    piece = PROMOTION_PIECES[promo % 4]

    dx = direction - 1
    tx, ty = fx + dx, fy + 1  # assumes white (fine for encoding)

    return chess.Move(from_sq, coords_to_square(tx, ty), promotion=piece)