import os
import sys
import time
from typing import List, Optional, Tuple

import chess
import pygame
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.chess_model import ChessModel
from bot.test_bot_connection import predict as nn_predict


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(REPO_ROOT, "outputs/models/latest.pt")

BOARD_SIZE = 640
PANEL_WIDTH = 320
WINDOW_W = BOARD_SIZE + PANEL_WIDTH
WINDOW_H = BOARD_SIZE
SQ = BOARD_SIZE // 8

LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
HIGHLIGHT = (120, 200, 120)
LAST_MOVE_HIGHLIGHT = (245, 245, 105)
LEGAL_DOT = (40, 40, 40)
BG = (28, 28, 28)
TEXT = (230, 230, 230)
ACCENT = (120, 170, 255)
WARN = (255, 120, 120)

PIECE_TO_GLYPH = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}


def square_to_xy(square: int, user_side: chess.Color) -> Tuple[int, int]:
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    if user_side == chess.WHITE:
        x = file_idx * SQ
        y = (7 - rank_idx) * SQ
    else:
        x = (7 - file_idx) * SQ
        y = rank_idx * SQ
    return x, y


def xy_to_square(x: int, y: int, user_side: chess.Color) -> Optional[int]:
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        return None
    cx = x // SQ
    cy = y // SQ
    if user_side == chess.WHITE:
        file_idx = cx
        rank_idx = 7 - cy
    else:
        file_idx = 7 - cx
        rank_idx = cy
    return chess.square(file_idx, rank_idx)


def load_model(path: str) -> ChessModel:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    model, _, _, _ = ChessModel.load_checkpoint(path, device=DEVICE)
    model.eval()
    return model


def choose_side_screen(screen, font, small_font) -> chess.Color:
    clock = pygame.time.Clock()
    white_btn = pygame.Rect(BOARD_SIZE // 2 - 190, BOARD_SIZE // 2 - 35, 160, 70)
    black_btn = pygame.Rect(BOARD_SIZE // 2 + 30, BOARD_SIZE // 2 - 35, 160, 70)
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if white_btn.collidepoint(ev.pos):
                    return chess.WHITE
                if black_btn.collidepoint(ev.pos):
                    return chess.BLACK

        screen.fill(BG)
        title = font.render("Play vs Your NN", True, TEXT)
        help_text = small_font.render("Choose your side", True, TEXT)
        pygame.draw.rect(screen, ACCENT, white_btn, border_radius=10)
        pygame.draw.rect(screen, ACCENT, black_btn, border_radius=10)
        screen.blit(title, (BOARD_SIZE // 2 - title.get_width() // 2, BOARD_SIZE // 2 - 120))
        screen.blit(help_text, (BOARD_SIZE // 2 - help_text.get_width() // 2, BOARD_SIZE // 2 - 75))
        wtxt = small_font.render("White", True, BG)
        btxt = small_font.render("Black", True, BG)
        screen.blit(wtxt, (white_btn.centerx - wtxt.get_width() // 2, white_btn.centery - wtxt.get_height() // 2))
        screen.blit(btxt, (black_btn.centerx - btxt.get_width() // 2, black_btn.centery - btxt.get_height() // 2))
        pygame.display.flip()
        clock.tick(60)


def draw_coords(screen, small_font, user_side: chess.Color):
    files = "abcdefgh"
    for i in range(8):
        if user_side == chess.WHITE:
            f = files[i]
            r = str(8 - i)
            fx = i * SQ + SQ - 14
            fy = BOARD_SIZE - 18
            rx = 4
            ry = i * SQ + 2
        else:
            f = files[7 - i]
            r = str(i + 1)
            fx = i * SQ + SQ - 14
            fy = BOARD_SIZE - 18
            rx = 4
            ry = i * SQ + 2
        screen.blit(small_font.render(f, True, TEXT), (fx, fy))
        screen.blit(small_font.render(r, True, TEXT), (rx, ry))


def draw_board(
    screen,
    board: chess.Board,
    user_side: chess.Color,
    piece_font,
    small_font,
    dragging_from: Optional[int],
    legal_targets: List[int],
    drag_mouse_pos: Optional[Tuple[int, int]],
    last_move: Optional[chess.Move],
):
    for rank in range(8):
        for file_idx in range(8):
            if user_side == chess.WHITE:
                sq = chess.square(file_idx, 7 - rank)
            else:
                sq = chess.square(7 - file_idx, rank)
            x = file_idx * SQ
            y = rank * SQ
            color = LIGHT if (file_idx + rank) % 2 == 0 else DARK
            pygame.draw.rect(screen, color, (x, y, SQ, SQ))
            if last_move is not None and (sq == last_move.from_square or sq == last_move.to_square):
                overlay = pygame.Surface((SQ, SQ), pygame.SRCALPHA)
                overlay.fill((*LAST_MOVE_HIGHLIGHT, 100))
                screen.blit(overlay, (x, y))
            if dragging_from is not None and sq == dragging_from:
                pygame.draw.rect(screen, HIGHLIGHT, (x, y, SQ, SQ), 4)
            if sq in legal_targets:
                pygame.draw.circle(screen, LEGAL_DOT, (x + SQ // 2, y + SQ // 2), 9)

            piece = board.piece_at(sq)
            if piece is not None and sq != dragging_from:
                glyph = PIECE_TO_GLYPH[piece.symbol()]
                t = piece_font.render(glyph, True, (20, 20, 20) if piece.color == chess.BLACK else (245, 245, 245))
                screen.blit(t, (x + SQ // 2 - t.get_width() // 2, y + SQ // 2 - t.get_height() // 2 + 2))

    if dragging_from is not None and drag_mouse_pos is not None:
        piece = board.piece_at(dragging_from)
        if piece is not None:
            glyph = PIECE_TO_GLYPH[piece.symbol()]
            t = piece_font.render(glyph, True, (20, 20, 20) if piece.color == chess.BLACK else (245, 245, 245))
            mx, my = drag_mouse_pos
            screen.blit(t, (mx - t.get_width() // 2, my - t.get_height() // 2))

    draw_coords(screen, small_font, user_side)


def draw_panel(
    screen,
    board: chess.Board,
    user_side: chess.Color,
    font,
    small_font,
    move_history: List[str],
    status_text: str,
    check_text: str,
):
    px = BOARD_SIZE + 16
    pygame.draw.rect(screen, BG, (BOARD_SIZE, 0, PANEL_WIDTH, WINDOW_H))
    side_label = "White" if user_side == chess.WHITE else "Black"
    turn_label = "White" if board.turn == chess.WHITE else "Black"
    screen.blit(font.render("VS Neural Net", True, TEXT), (px, 12))
    screen.blit(small_font.render(f"You: {side_label}", True, TEXT), (px, 54))
    screen.blit(small_font.render(f"Turn: {turn_label}", True, TEXT), (px, 76))
    screen.blit(small_font.render(f"Fullmove #: {board.fullmove_number}", True, TEXT), (px, 98))
    screen.blit(small_font.render("Moves:", True, ACCENT), (px, 130))

    y = 154
    for i, mv in enumerate(move_history[-18:], 1):
        screen.blit(small_font.render(mv, True, TEXT), (px, y))
        y += 20

    color = WARN if "Illegal" in status_text or "error" in status_text.lower() else TEXT
    screen.blit(small_font.render(status_text, True, color), (px, WINDOW_H - 30))
    if check_text:
        screen.blit(small_font.render(check_text, True, WARN), (px, WINDOW_H - 56))


def main():
    pygame.init()
    pygame.display.set_caption("Play vs NN")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    font = pygame.font.SysFont("DejaVu Sans", 30, bold=True)
    piece_font = pygame.font.SysFont("DejaVu Sans", 52)
    small_font = pygame.font.SysFont("DejaVu Sans", 20)
    clock = pygame.time.Clock()

    model = load_model(MODEL_PATH)
    user_side = choose_side_screen(screen, font, small_font)
    nn_side = chess.BLACK if user_side == chess.WHITE else chess.WHITE
    board = chess.Board()

    dragging_from = None
    drag_mouse_pos = None
    legal_targets: List[int] = []
    move_history: List[str] = []
    status_text = "Drag your piece and drop to a target square."
    last_move: Optional[chess.Move] = None

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if board.is_game_over(claim_draw=True):
                continue

            if board.turn == user_side:
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    sq = xy_to_square(*ev.pos, user_side)
                    if sq is None:
                        continue
                    piece = board.piece_at(sq)
                    if piece is None:
                        status_text = "No piece on that square."
                        continue
                    if piece.color != user_side:
                        status_text = "That's not your piece."
                        continue
                    dragging_from = sq
                    drag_mouse_pos = ev.pos
                    legal_targets = [m.to_square for m in board.legal_moves if m.from_square == sq]

                elif ev.type == pygame.MOUSEMOTION and dragging_from is not None:
                    drag_mouse_pos = ev.pos

                elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1 and dragging_from is not None:
                    to_sq = xy_to_square(*ev.pos, user_side)
                    moved = False
                    if to_sq is not None:
                        # Promotion default to queen if needed.
                        move = chess.Move(dragging_from, to_sq)
                        if board.piece_at(dragging_from) and board.piece_at(dragging_from).piece_type == chess.PAWN:
                            rank = chess.square_rank(to_sq)
                            if rank in (0, 7):
                                move = chess.Move(dragging_from, to_sq, promotion=chess.QUEEN)
                        if move in board.legal_moves:
                            san = board.san(move)
                            board.push(move)
                            last_move = move
                            move_history.append(f"{board.fullmove_number - (0 if board.turn == chess.WHITE else 1)}. {san}")
                            status_text = f"You played {move.uci()}"
                            moved = True
                        else:
                            status_text = "Illegal move."
                    else:
                        status_text = "Drop inside board."

                    dragging_from = None
                    drag_mouse_pos = None
                    legal_targets = []
                    if moved:
                        pass

        # NN turn
        if running and not board.is_game_over(claim_draw=True) and board.turn == nn_side:
            start = time.time()
            move_uci, _, _ = nn_predict(board.fen(), model)
            think_ms = int((time.time() - start) * 1000)
            if move_uci is None:
                status_text = "NN has no legal move."
            else:
                mv = chess.Move.from_uci(move_uci)
                if mv not in board.legal_moves:
                    mv = next(iter(board.legal_moves))
                san = board.san(mv)
                board.push(mv)
                last_move = mv
                move_history.append(f"{board.fullmove_number - (0 if board.turn == chess.WHITE else 1)}... {san}")
                status_text = f"NN played {mv.uci()} ({think_ms} ms)"

        # Draw
        screen.fill(BG)
        check_text = "Check!" if board.is_check() else ""
        draw_board(
            screen,
            board,
            user_side,
            piece_font,
            small_font,
            dragging_from,
            legal_targets,
            drag_mouse_pos,
            last_move,
        )
        draw_panel(screen, board, user_side, font, small_font, move_history, status_text, check_text)

        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                result_text = f"Game over: {outcome.result()} ({outcome.termination.name})"
            else:
                result_text = "Game over"
            overlay = small_font.render(result_text, True, WARN)
            screen.blit(overlay, (BOARD_SIZE + 16, WINDOW_H - 56))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()

