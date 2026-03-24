import numpy as np

# Piece to channel mapping
PIECE_TO_CHANNEL = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}


def parse_fen(fen):
    """
    Splits FEN into components
    """
    parts = fen.split(" ")
    board_part = parts[0]
    turn = parts[1]
    castling = parts[2]
    return board_part, turn, castling


def board_to_matrix(board_part):
    """
    Converts FEN board into 8x8 matrix
    """
    board = []
    rows = board_part.split("/")

    for row in rows:
        current_row = []
        for char in row:
            if char.isdigit():
                current_row.extend(['.'] * int(char))
            else:
                current_row.append(char)
        board.append(current_row)

    return board  # shape (8,8)


def fen_to_tensor(fen):
    """
    Main function: FEN → (20,8,8) tensor
    """
    board_part, turn, castling = parse_fen(fen)
    board = board_to_matrix(board_part)

    tensor = np.zeros((20, 8, 8), dtype=np.float32)

    # -------------------------
    # 1. Piece placement (12 channels)
    # -------------------------
    for i in range(8):
        for j in range(8):
            piece = board[i][j]
            if piece != '.':
                channel = PIECE_TO_CHANNEL[piece]
                tensor[channel][i][j] = 1

    # -------------------------
    # 2. Side to move (1 channel)
    # -------------------------
    if turn == 'w':
        tensor[12][:, :] = 1
    else:
        tensor[13][:, :] = 1

    # -------------------------
    # 3. Castling rights (4 channels)
    # -------------------------
    if 'K' in castling:
        tensor[14][:, :] = 1
    if 'Q' in castling:
        tensor[15][:, :] = 1
    if 'k' in castling:
        tensor[16][:, :] = 1
    if 'q' in castling:
        tensor[17][:, :] = 1

    # -------------------------
    # 4. Empty / reserved channels
    # -------------------------
    # Can use for move count, repetition, etc.
    # Keeping as zeros for now

    return tensor