START
  │
  ▼
Collect Chess Games
(Lichess PGN dataset)
  │
  ▼
Parse PGN Files
Extract:
- game_id
- move_number
- FEN
- human_move
  │
  ▼
Stockfish Evaluation
For each position:
- best_move
- eval_before
- eval_after
  │
  ▼
Create Dataset
games.csv
  │
  ▼
────────────────────────
PHASE 1 — DATA ENCODING
────────────────────────
  │
  ▼
Board Encoding
FEN → (20,8,8) tensor
  │
  ▼
Policy Encoding
move → index (0–4671)
  │
  ▼
Legal Move Mask
Generate legal moves
mask illegal moves
  │
  ▼
Encoded Dataset
encoded_data/
  │
  ▼
────────────────────────
PHASE 2 — DATA PIPELINE
────────────────────────
  │
  ▼
Dataset Sharding
Split into:
shard_0.pt
shard_1.pt
shard_2.pt
  │
  ▼
PyTorch Dataset Class
(chess_dataset.py)
  │
  ▼
DataLoader
Batch loading
  │
  ▼
────────────────────────
PHASE 3 — NEURAL NETWORK
────────────────────────
  │
  ▼
Input Tensor
(20,8,8)
  │
  ▼
Convolution Layer
Feature Extraction
  │
  ▼
Residual Blocks
Learn chess patterns
  │
  ▼
Split into Heads
  │
  ├── Policy Head → Move probabilities (4672)
  │
  └── Value Head → Position evaluation (-1 to +1)
  │
  ▼
────────────────────────
PHASE 4 — TRAINING
────────────────────────
  │
  ▼
Forward Pass
Model predicts:
- move probabilities
- evaluation
  │
  ▼
Loss Calculation
Policy Loss  → CrossEntropy
Value Loss   → MSE
  │
  ▼
Total Loss
loss = policy_loss + value_loss
  │
  ▼
Backpropagation
Update model weights
  │
  ▼
Checkpoint Saving
model_epoch_1.pt
model_epoch_2.pt
  │
  ▼
────────────────────────
PHASE 5 — MODEL EVALUATION
────────────────────────
  │
  ▼
Evaluation Metrics
- policy accuracy
- top-3 accuracy
- value prediction error
- blunder detection accuracy
  │
  ▼
────────────────────────
PHASE 6 — CHESS ENGINE
────────────────────────
  │
  ▼
Monte Carlo Tree Search
(MCTS)
  │
  ▼
Neural Network Evaluation
for each node
  │
  ▼
Best Move Selection
  │
  ▼
────────────────────────
PHASE 7 — FINAL TOOL
────────────────────────
  │
  ▼
Input: FEN Position
  │
  ▼
Model + MCTS
  │
  ▼
Output:
- Best Move
- Top 3 Moves
- Evaluation
- Blunder Warning
  │
  ▼
END