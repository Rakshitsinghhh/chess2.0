import { Chess } from "chess.js";

import { fenToInputTensorData } from "../chess/fenTensor";
import { moveToPolicyIndex, TOTAL_MOVES } from "../chess/movePolicy";
import { getOnnxSession, getOrtModule } from "./session";

export type OnnxPredictResult = {
  move_uci: string | null;
  model_score: number;
  top_candidates: string[];
  turn: "w" | "b";
  fen: string;
};

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

function terminalValue(chess: Chess): number | null {
  if (chess.isCheckmate()) {
    const t = chess.turn();
    return t === "b" ? 1.0 : -1.0;
  }
  if (chess.isStalemate()) {
    return 0.0;
  }
  return null;
}

export async function onnxFastPredict(fen: string): Promise<OnnxPredictResult> {
  const chess = new Chess(fen);
  const turn: "w" | "b" = chess.turn();

  const tv = terminalValue(chess);
  if (tv !== null) {
    return {
      move_uci: null,
      model_score: tv,
      top_candidates: [],
      turn,
      fen
    };
  }

  const legalMoves = chess.moves({ verbose: true });
  const legalUcis = new Set<string>();
  const indexToLegalUci = new Map<number, string>();

  for (const m of legalMoves) {
    const promo = m.promotion as string | undefined;
    try {
      const idx = moveToPolicyIndex(m.from, m.to, promo);
      const uci = m.from + m.to + (promo ?? "");
      legalUcis.add(uci);
      indexToLegalUci.set(idx, uci);
    } catch {
      continue;
    }
  }

  if (legalMoves.length > 0 && indexToLegalUci.size === 0) {
    throw new Error("No legal moves could be encoded (policy index mismatch)");
  }

  const ort = await getOrtModule();
  const session = await getOnnxSession();
  const inputData = fenToInputTensorData(fen);
  const input = new ort.Tensor("float32", inputData, [1, 20, 8, 8]);
  const feeds = { board: input };
  const out = await session.run(feeds);

  const policy = out.policy_logits.data as Float32Array;
  const valueArr = out.value.data as Float32Array;
  const rawValue = valueArr[0] ?? 0;

  const mask = new Float32Array(TOTAL_MOVES);
  for (const idx of indexToLegalUci.keys()) {
    mask[idx] = 1;
  }

  let bestIdx = 0;
  let bestScore = -Infinity;
  for (const idx of indexToLegalUci.keys()) {
    const s = policy[idx] + (mask[idx] - 1) * 1e9;
    if (s > bestScore) {
      bestScore = s;
      bestIdx = idx;
    }
  }

  let bestUci = indexToLegalUci.get(bestIdx) ?? null;
  if (!bestUci && legalMoves.length > 0) {
    const fallback = legalMoves[0];
    bestUci = fallback.from + fallback.to + ((fallback.promotion as string) ?? "");
  }

  if (bestUci && !legalUcis.has(bestUci)) {
    let maxL = -Infinity;
    let pick = -1;
    for (let i = 0; i < TOTAL_MOVES; i++) {
      if (!indexToLegalUci.has(i)) continue;
      if (policy[i] > maxL) {
        maxL = policy[i];
        pick = i;
      }
    }
    if (pick >= 0) {
      bestIdx = pick;
      bestUci = indexToLegalUci.get(pick) ?? bestUci;
    }
  }

  const topCandidates: string[] = [];
  const scored: { idx: number; logit: number }[] = [];
  for (const idx of indexToLegalUci.keys()) {
    scored.push({ idx, logit: policy[idx] });
  }
  scored.sort((a, b) => b.logit - a.logit);
  const seen = new Set<string>();
  for (const { idx } of scored) {
    const uci = indexToLegalUci.get(idx);
    if (uci && !seen.has(uci)) {
      seen.add(uci);
      topCandidates.push(uci);
    }
    if (topCandidates.length >= 5) break;
  }

  return {
    move_uci: bestUci,
    model_score: clamp(rawValue, -1, 1),
    top_candidates: topCandidates,
    turn,
    fen
  };
}

export async function onnxEvalValue(fen: string): Promise<{ model_score: number; turn: "w" | "b"; fen: string }> {
  const r = await onnxFastPredict(fen);
  return { model_score: r.model_score, turn: r.turn, fen: r.fen };
}
