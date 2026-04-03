import os
import sys
from typing import List, Optional

import chess
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.chess_model import ChessModel
from bot.test_bot_connection import predict as nn_predict
from utils.fen_utils import fen_to_tensor
from utils.generate_move_mask import generate_move_mask
from utils.move_index_encoding import policy_index_to_move, move_to_policy_index


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(REPO_ROOT, "outputs/models/latest.pt"))
FAST_MODE = os.environ.get("FAST_MODE", "1") == "1"

app = FastAPI(title="Chess2.0 Model Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

model: Optional[ChessModel] = None


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


@torch.inference_mode()
def fast_eval_value(fen: str, model_obj: ChessModel) -> float:
    """Single forward pass; value only. Same terminal handling as fast_predict."""
    board = chess.Board(fen)
    if board.is_checkmate():
        return 1.0 if board.turn == chess.BLACK else -1.0
    if board.is_stalemate():
        return 0.0

    board_tensor = torch.tensor(fen_to_tensor(fen), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    _, value = model_obj(board_tensor)
    return float(torch.clamp(value, -1.0, 1.0).item())


@torch.inference_mode()
def fast_predict(fen: str, model_obj: ChessModel):
    """
    Faster inference path for web usage:
    - single forward pass
    - legal move masking
    - argmax selection + top candidates
    """
    board = chess.Board(fen)
    if board.is_checkmate():
        return None, (1.0 if board.turn == chess.BLACK else -1.0), []
    if board.is_stalemate():
        return None, 0.0, []

    board_tensor = torch.tensor(fen_to_tensor(fen), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    mask = torch.tensor(generate_move_mask(fen), dtype=torch.float32).to(DEVICE)
    policy_logits, value = model_obj(board_tensor)
    masked_logits = policy_logits.squeeze(0) + (mask - 1) * 1e9

    legal_ucis = [m.uci() for m in board.legal_moves]
    best_idx = torch.argmax(masked_logits).item()
    try:
        best_uci = policy_index_to_move(best_idx).uci()
    except Exception:
        best_uci = legal_ucis[0]
    if best_uci not in legal_ucis:
        # fallback: best among legal encodable moves
        logits_np = masked_logits.cpu().numpy()
        best_score = -float("inf")
        best_uci = legal_ucis[0]
        for lm in board.legal_moves:
            try:
                idx = move_to_policy_index(lm)
            except Exception:
                continue
            if logits_np[idx] > best_score:
                best_score = logits_np[idx]
                best_uci = lm.uci()

    # collect top candidates
    top_indices = torch.topk(masked_logits, min(20, masked_logits.shape[0])).indices.tolist()
    top_candidates: List[str] = []
    seen = set()
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

    return best_uci, float(torch.clamp(value, -1.0, 1.0).item()), top_candidates


@app.on_event("startup")
def startup() -> None:
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    model, _, _, _ = ChessModel.load_checkpoint(MODEL_PATH, device=DEVICE)
    model.eval()


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "model_path": MODEL_PATH, "fast_mode": FAST_MODE}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        board = chess.Board(req.fen)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {e}")

    try:
        if FAST_MODE:
            move_uci, score, top5 = fast_predict(req.fen, model)
        else:
            move_uci, score, top5 = nn_predict(req.fen, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return PredictResponse(
        move_uci=move_uci,
        model_score=float(score),
        top_candidates=top5[:5] if top5 else [],
        turn="w" if board.turn == chess.WHITE else "b",
        fen=req.fen
    )


@app.post("/eval", response_model=EvalResponse)
def eval_position(req: EvalRequest):
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        board = chess.Board(req.fen)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {e}")

    try:
        score = fast_eval_value(req.fen, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return EvalResponse(
        model_score=float(score),
        turn="w" if board.turn == chess.WHITE else "b",
        fen=req.fen,
    )

