#!/usr/bin/env python3
"""
Export ChessModel checkpoint to ONNX for Next.js / onnxruntime-node.

Writes: nextjs_host/public/model.onnx

Usage (repo root):
  . .venv/bin/activate
  python scripts/export_model_onnx.py
  python scripts/export_model_onnx.py --checkpoint outputs/models/latest.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NEXT_PUBLIC = REPO_ROOT / "nextjs_host" / "public"
OUT_DEFAULT = NEXT_PUBLIC / "model.onnx"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from model.chess_model import ChessModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("MODEL_PATH", str(REPO_ROOT / "outputs/models/latest.pt")),
        help="Path to .pt checkpoint",
    )
    parser.add_argument("--out", default=str(OUT_DEFAULT), help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    model, _, _, _ = ChessModel.load_checkpoint(str(ckpt), device=device)
    model.eval()

    dummy = torch.randn(1, 20, 8, 8, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["board"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "board": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=args.opset,
        export_params=True,
        do_constant_folding=True,
    )

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
