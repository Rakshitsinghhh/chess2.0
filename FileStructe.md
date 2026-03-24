chess-blunder-detection/
│
├── data/
│   ├── raw/
│   │   └── games.csv/
│   │
│   ├── processed/
|   |   └── stockfish_labeled_moves.csv/
│   │
│   └── shards/
│       ├── shard_000.pt/
│       ├── shard_001.pt/
│       └── shard_002.pt/
│
│
├── preprocessing/
│   ├── label_with_stockfish.py/
│   ├── create_shards.py/
│   └── split_dataset.py/
│
│
├── dataset/
│   ├── download_dataset.py/
│   ├── chess_dataset.py/
│   └── dataloader.py/
│
│
├── models/
│   ├── cnn_encoder.py/
│   ├── residual_block.py/
│   ├── policy_head.py/
│   ├── value_head.py/
│   └── chess_model.py/
│
│
├── training/
│   ├── train.py/
│   ├── trainer.py/
│   └── loss.py/
│
│
├── inference/
│   ├── predict_move.py/
│   ├── evaluate_position.py/
│   └── mcts.py/
│
│
├── utils/
│   ├── fen_utils.py/
│   ├── move-index_encoding.py/
│   └── generate_move_mask.py/
│
│
├── outputs/
│   ├── models/
│   │   └── chess_model.pt/
│   │
│   ├── logs/
│   │   └── training.log/
│   │
│   └── plots/
│       └── loss_curve.png/
│
│
├── requirements.txt/
└── README.md /