# Play Against the Neural Network

This folder lets a user play chess against your trained model in two modes:

- `play_vs_nn.py` -> terminal/CLI mode
- `play_vs_nn_ui.py` -> drag-and-drop UI mode

Both modes use your project model checkpoint (default: `outputs/models/latest.pt`).

---

## Quick Start (Clone + Run)

### 1) Clone the repository

```bash
git clone <your-repo-url>
cd chess2.0
```

### 2) Create and activate Python environment

```bash
python3 -m venv .venv
. .venv/bin/activate
```

### 3) Install dependencies

At minimum, make sure these are installed:

```bash
python3 -m pip install torch python-chess tqdm pygame
```

If your project already has a requirements file, use that instead:

```bash
python3 -m pip install -r requirements.txt
```

### 4) Make sure a model checkpoint exists

Expected default checkpoint:

```text
outputs/models/latest.pt
```

If you want to use another checkpoint, set:

```bash
MODEL_PATH=/absolute/path/to/model.pt
```

---

## Run the Game

### Option A: Terminal mode

```bash
. .venv/bin/activate
python3 play_vs_user_nn/play_vs_nn.py
```

### Option B: UI mode (drag and drop)

```bash
. .venv/bin/activate
python3 play_vs_user_nn/play_vs_nn_ui.py
```

UI features:

- choose White or Black
- drag-and-drop pieces
- board coordinates (`a-h`, `1-8`)
- last-move highlight
- check indicator
- move list + move count

---

## Web app (Vercel / Next.js)

The browser UI and API routes live under **`nextjs_host/`**. Full instructions: [`nextjs_host/README.md`](nextjs_host/README.md).

**Vercel:** set the project **Root Directory** to **`nextjs_host`** (not the repo root). Then configure `USE_ONNX` / `MODEL_ONNX_URL` or `MODEL_SERVER_URL` as described in that README.

---

## How the Model Chooses Moves

This game uses your **neural network**, not Stockfish, for move decisions.

High-level flow:

1. Board position (`FEN`) is converted to model tensor input.
2. Model outputs:
   - **policy logits** (which move to play)
   - **value** in `[-1, 1]` (position score)
3. Illegal moves are masked out.
4. The best legal candidate is selected using your current `predict()` logic (policy + value reranking + mate/stalemate safeguards).

In short: the network is acting as both the move selector and evaluator.

---

## Technology Stack

- **PyTorch**: neural network inference
- **python-chess**: board rules, legal move generation, game state
- **pygame**: drag-and-drop chess UI
- **tqdm / standard Python**: utilities used in the project

---

## Input Format Notes (CLI mode)

- Use **UCI** move format, for example:
  - `e2e4`
  - `g1f3`
  - `a7a8q` (promotion)
- If you type short algebraic like `e4`, CLI mode will reject it and ask for full UCI.

---

## Output Files

CLI mode saves PGN games to:

```text
outputs/pgn/
```

Example:

```text
outputs/pgn/nn_game_YYYYMMDD_HHMMSS.pgn
```

---

## Troubleshooting

- `ModuleNotFoundError: model`
  - Run from repo root or use:
    - `cd /path/to/chess2.0`
    - `. .venv/bin/activate`
    - `python3 play_vs_user_nn/play_vs_nn.py`
- `ModuleNotFoundError: torch` or `pygame`
  - Install missing package(s) inside your `.venv`.
- `Model not found`
  - Ensure `outputs/models/latest.pt` exists or set `MODEL_PATH`.

