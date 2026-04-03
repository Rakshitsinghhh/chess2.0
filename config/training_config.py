# config/training_config.py
"""Centralized training configuration"""

REGULAR_CONFIG = {
    "shard_dir": "data/shards",
    "output_dir": "outputs/models",
    "batch_size": 64,
    "accumulation_steps": 4,
    "epochs_per_shard": 2,
    "passes": 5,
    "learning_rate": 1e-3,
    "value_weight": 0.05,
    "label_smoothing": 0.1,
    "grad_clip": 1.0,
}

ENDGAME_CONFIG = {
    "enabled": True,
    "endgame_shard_dir": "data/endgame_shards",
    "endgame_weight": 2.0,
    "batch_ratio": 0.25,
    "epochs": 10,
    "learning_rate": 5e-4,
    "accumulation_steps": 4,
    "value_weight": 0.05,
    "label_smoothing": 0.1,
    "num_workers": 0,
}

COMMON_CONFIG = {
    "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
    "input_channels": 20,
    "num_moves": 4672,
    "num_workers": 2,
}