# Play vs User (NN)

Interactive command-line match between you and your neural-network chess model.

## How to run

```bash
. .venv/bin/activate
python3 play_vs_user_nn/play_vs_nn.py
```

## UI version (drag and drop)

```bash
. .venv/bin/activate
python3 play_vs_user_nn/play_vs_nn_ui.py
```

If `pygame` is missing:

```bash
python3 -m pip install pygame
```

## What it does

1. Ask you to choose `white` or `black`
2. Starts from the initial chess position
3. On your turn: you enter a move in **UCI** format (example: `e2e4`, `g1f3`, `a7a8q`)
4. On the NN turn: it prints the chosen move and shows:
   - NN value (from your viewpoint)
   - policy probability for the chosen move
   - top candidate moves (optional, limited)
5. Game ends on checkmate, stalemate, or other PGN outcome.

## Notes

- This uses your neural network only (no Stockfish).
- The NN move selection uses the same `predict()` logic that you use for comparison (policy candidates + value reranking + terminal overrides).

