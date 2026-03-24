import pandas as pd
import chess
import chess.engine
import time
import sys
from datetime import timedelta

STOCKFISH_PATH = "/usr/games/stockfish"  # change if needed
ENGINE_DEPTH = 12

# ---------- progress display ----------

class ProgressTracker:
    def __init__(self, total_games, total_moves_estimate=None):
        self.total_games = total_games
        self.total_moves_estimate = total_moves_estimate
        self.start_time = time.time()
        self.games_done = 0
        self.moves_done = 0
        self.blunders = 0
        self.mistakes = 0
        self.inaccuracies = 0
        self.best_moves = 0
        self._bar_width = 40

    def _elapsed(self):
        return time.time() - self.start_time

    def _eta(self):
        elapsed = self._elapsed()
        if self.games_done == 0:
            return "calculating..."
        rate = self.games_done / elapsed          # games per second
        remaining = self.total_games - self.games_done
        eta_secs = remaining / rate if rate > 0 else 0
        return str(timedelta(seconds=int(eta_secs)))

    def _bar(self, fraction):
        filled = int(self._bar_width * fraction)
        bar = "█" * filled + "░" * (self._bar_width - filled)
        return bar

    def _move_rate(self):
        elapsed = self._elapsed()
        if elapsed == 0:
            return 0.0
        return self.moves_done / elapsed

    def update(self, game_index, move_number, label, total_moves_this_game):
        self.moves_done += 1
        if label == "best":        self.best_moves  += 1
        elif label == "inaccuracy":self.inaccuracies+= 1
        elif label == "mistake":   self.mistakes    += 1
        elif label == "blunder":   self.blunders    += 1

        game_fraction  = (game_index) / self.total_games
        move_fraction  = move_number  / max(total_moves_this_game, 1)
        overall_frac   = min((game_index + move_fraction) / self.total_games, 1.0)

        elapsed_str = str(timedelta(seconds=int(self._elapsed())))
        rate        = self._move_rate()

        # ── build the display ─────────────────────────────────────────────
        lines = [
            "",
            f"  ╔══════════════════════════════════════════════════════╗",
            f"  ║          Chess Dataset Labeler — Progress            ║",
            f"  ╠══════════════════════════════════════════════════════╣",
            f"  ║  Overall   [{self._bar(overall_frac)}] {overall_frac*100:5.1f}%  ║",
            f"  ║  Game      [{self._bar(game_fraction)}] {game_fraction*100:5.1f}%  ║",
            f"  ║  This game [{self._bar(move_fraction)}] {move_fraction*100:5.1f}%  ║",
            f"  ╠══════════════════════════════════════════════════════╣",
            f"  ║  Games  : {self.games_done:>5} / {self.total_games:<5}   Elapsed : {elapsed_str:<12}  ║",
            f"  ║  Moves  : {self.moves_done:>7}            ETA     : {self._eta():<12}  ║",
            f"  ║  Rate   : {rate:>6.1f} moves/s       Game #  : {game_index+1:<5}        ║",
            f"  ╠══════════════════════════════════════════════════════╣",
            f"  ║  Move labels so far:                                 ║",
            f"  ║    ✓ Best       : {self.best_moves:<8}                          ║",
            f"  ║    ~ Inaccuracy : {self.inaccuracies:<8}                          ║",
            f"  ║    ! Mistake    : {self.mistakes:<8}                          ║",
            f"  ║    ✗ Blunder    : {self.blunders:<8}                          ║",
            f"  ╚══════════════════════════════════════════════════════╝",
        ]

        # move cursor up to overwrite previous block (18 lines + 1 blank)
        if self.moves_done > 1 or self.games_done > 0:
            sys.stdout.write(f"\033[{len(lines)}A")  # ANSI: move cursor up N lines

        print("\n".join(lines), flush=True)

    def finish_game(self):
        self.games_done += 1

    def done(self, total_rows, output_csv):
        elapsed_str = str(timedelta(seconds=int(self._elapsed())))
        rate        = self._move_rate()
        print(
            f"\n  ✅  Done!  {total_rows:,} labeled moves → {output_csv}\n"
            f"      Total time : {elapsed_str}  |  Avg rate : {rate:.1f} moves/s\n"
        )


# ---------- helper functions ----------

def get_eval(engine, board):
    info = engine.analyse(board, chess.engine.Limit(depth=ENGINE_DEPTH))
    score = info["score"].white().score(mate_score=10000)
    best_move = info["pv"][0].uci()
    return score, best_move


def classify_move(eval_loss):
    if eval_loss < 0.2:
        return "best"
    elif eval_loss < 0.5:
        return "inaccuracy"
    elif eval_loss < 1.5:
        return "mistake"
    else:
        return "blunder"


# ---------- main pipeline ----------

def label_chess_dataset(input_csv, output_csv, max_games=None):
    df = pd.read_csv(input_csv)
    if max_games:
        df = df.head(max_games)

    total_games = len(df)
    # rough estimate: average ~40 moves per game
    tracker = ProgressTracker(total_games, total_moves_estimate=total_games * 40)

    print(f"\n  Starting — {total_games} games to process …\n")

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    labeled_rows = []

    for idx, row in df.iterrows():
        game_id   = row["id"]
        moves_str = row["moves"]

        if pd.isna(moves_str):
            tracker.finish_game()
            continue

        board = chess.Board()
        moves = moves_str.split()
        total_moves_this_game = len(moves)

        for move_number, san in enumerate(moves, start=1):
            try:
                fen_before = board.fen()

                eval_before, engine_best = get_eval(engine, board)

                human_move = board.parse_san(san)
                board.push(human_move)

                eval_after, _ = get_eval(engine, board)

                eval_loss = abs(eval_before - eval_after)
                label     = classify_move(eval_loss)

                labeled_rows.append({
                    "game_id":       game_id,
                    "move_number":   move_number,
                    "fen":           fen_before,
                    "human_move":    human_move.uci(),
                    "engine_best":   engine_best,
                    "eval_before":   eval_before,
                    "eval_after":    eval_after,
                    "eval_loss":     eval_loss,
                    "label":         label,
                    "white_rating":  row["white_rating"],
                    "black_rating":  row["black_rating"],
                    "winner":        row["winner"],
                    "opening":       row["opening_name"],
                })

                # update display after every move
                tracker.update(idx, move_number, label, total_moves_this_game)

            except Exception:
                # illegal SAN, corrupted move, etc.
                break

        tracker.finish_game()

    engine.quit()

    out_df = pd.DataFrame(labeled_rows)
    out_df.to_csv(output_csv, index=False)
    tracker.done(len(out_df), output_csv)


# ---------- run ----------

label_chess_dataset(
    input_csv="games.csv",
    output_csv="labeled_chess_moves.csv",
    # max_games=100  # start small
)