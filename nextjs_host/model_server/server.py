"""
Chess2.0 — FastAPI Model Server
Run:  uvicorn server:app --host 0.0.0.0 --port 8001
"""
import os
import sys
import urllib.request
from contextlib import asynccontextmanager
from typing import List, Optional

import chess
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel


# ─── Repo root on sys.path ───────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.chess_model import ChessModel          # noqa: E402
from inference.predict import predict as nn_predict  # noqa: E402
from utils.fen_utils import fen_to_tensor          # noqa: E402
from utils.generate_move_mask import generate_move_mask  # noqa: E402
from utils.move_index_encoding import (            # noqa: E402
    policy_index_to_move,
    move_to_policy_index,
)


# ─── Config ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(REPO_ROOT, "outputs/models/latest.pt"),
)
# Optional: set MODEL_URL to a public HTTPS URL and the server will download
# the weights at startup (useful for Railway / Render where you can't commit a
# 254 MB file to git).  Example:
#   MODEL_URL=https://huggingface.co/your-user/chess2.0/resolve/main/latest.pt
MODEL_URL = os.environ.get("MODEL_URL", "")
FAST_MODE = os.environ.get("FAST_MODE", "1") == "1"


# ─── Lifespan (replaces deprecated @app.on_event) ────────────────────────────
_model: Optional[ChessModel] = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    global _model

    # Download weights if a URL is provided and the file is missing
    if MODEL_URL and not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print(f"[startup] Downloading model from {MODEL_URL} → {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[startup] Download complete.")

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}\n"
            "Set MODEL_PATH or MODEL_URL environment variables."
        )

    print(f"[startup] Loading model from {MODEL_PATH} on {DEVICE} …")
    _model, _, _, _ = ChessModel.load_checkpoint(MODEL_PATH, device=DEVICE)
    _model.eval()
    print("[startup] Model ready ✓")

    yield  # ← server is running

    _model = None
    print("[shutdown] Model unloaded.")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Chess2.0 Model Server",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ─────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    fen: str


class PredictResponse(BaseModel):
    move_uci: Optional[str]
    model_score: float
    top_candidates: List[str]
    turn: str
    fen: str


class EvalRequest(BaseModel):
    fen: str


class EvalResponse(BaseModel):
    model_score: float
    turn: str
    fen: str


# ─── Inference helpers ────────────────────────────────────────────────────────
@torch.inference_mode()
def fast_predict(fen: str, m: ChessModel):
    board = chess.Board(fen)
    if board.is_checkmate():
        return None, (1.0 if board.turn == chess.BLACK else -1.0), []
    if board.is_stalemate() or board.is_insufficient_material():
        return None, 0.0, []

    board_tensor = torch.tensor(
        fen_to_tensor(fen), dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)
    mask = torch.tensor(generate_move_mask(fen), dtype=torch.float32).to(DEVICE)

    policy_logits, value = m(board_tensor)
    masked_logits = policy_logits.squeeze(0) + (mask - 1) * 1e9

    legal_ucis = [mv.uci() for mv in board.legal_moves]
    best_idx = torch.argmax(masked_logits).item()
    try:
        best_uci = policy_index_to_move(best_idx).uci()
    except Exception:
        best_uci = legal_ucis[0] if legal_ucis else None

    if best_uci not in legal_ucis:
        logits_np = masked_logits.cpu().numpy()
        best_score = -float("inf")
        best_uci = legal_ucis[0] if legal_ucis else None
        for lm in board.legal_moves:
            try:
                idx = move_to_policy_index(lm)
            except Exception:
                continue
            if idx is not None and logits_np[idx] > best_score:
                best_score = logits_np[idx]
                best_uci = lm.uci()

    # top-5 candidates
    top_indices = torch.topk(
        masked_logits, min(20, masked_logits.shape[0])
    ).indices.tolist()
    top_candidates: List[str] = []
    seen: set = set()
    for idx in top_indices:
        try:
            uci = policy_index_to_move(idx).uci()
        except Exception:
            continue
        if uci in legal_ucis and uci not in seen:
            seen.add(uci)
            top_candidates.append(uci)
        if len(top_candidates) >= 5:
            break

    score = float(torch.clamp(value, -1.0, 1.0).item())
    return best_uci, score, top_candidates


@torch.inference_mode()
def fast_eval_value(fen: str, m: ChessModel) -> float:
    board = chess.Board(fen)
    if board.is_checkmate():
        return 1.0 if board.turn == chess.BLACK else -1.0
    if board.is_stalemate():
        return 0.0
    board_tensor = torch.tensor(
        fen_to_tensor(fen), dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)
    _, value = m(board_tensor)
    return float(torch.clamp(value, -1.0, 1.0).item())


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {
        "ok": _model is not None,
        "device": DEVICE,
        "model_path": MODEL_PATH,
        "fast_mode": FAST_MODE,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        chess.Board(req.fen)  # validate FEN
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {exc}")

    try:
        if FAST_MODE:
            move_uci, score, top5 = fast_predict(req.fen, _model)
        else:
            move_uci, score, top5 = nn_predict(req.fen, _model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    board = chess.Board(req.fen)
    return PredictResponse(
        move_uci=move_uci,
        model_score=float(score),
        top_candidates=top5[:5] if top5 else [],
        turn="w" if board.turn == chess.WHITE else "b",
        fen=req.fen,
    )


@app.post("/eval", response_model=EvalResponse)
def eval_position(req: EvalRequest):
    if _model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        board = chess.Board(req.fen)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {exc}")

    try:
        score = fast_eval_value(req.fen, _model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    return EvalResponse(
        model_score=float(score),
        turn="w" if board.turn == chess.WHITE else "b",
        fen=req.fen,
    )
