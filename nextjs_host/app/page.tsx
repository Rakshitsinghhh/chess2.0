"use client";

import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { Chess, Move, type Square as ChessSquare } from "chess.js";
import { Chessboard } from "react-chessboard";

type Side = "w" | "b";

type MoveResponse = {
  move_uci: string | null;
  model_score: number;
  top_candidates: string[];
  turn: Side;
  fen: string;
};

type PlyEval = { ply: number; score: number };

function kingSquare(g: Chess, color: Side): string | null {
  const b = g.board();
  const files = "abcdefgh";
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const p = b[r][f];
      if (p && p.type === "k" && p.color === color) {
        return `${files[f]}${8 - r}`;
      }
    }
  }
  return null;
}

async function mapLimit<T, R>(items: T[], limit: number, fn: (item: T, index: number) => Promise<R>): Promise<R[]> {
  const ret: R[] = new Array(items.length);
  let next = 0;
  async function worker() {
    for (;;) {
      const i = next++;
      if (i >= items.length) break;
      ret[i] = await fn(items[i], i);
    }
  }
  const n = Math.min(limit, Math.max(1, items.length));
  await Promise.all(Array.from({ length: n }, () => worker()));
  return ret;
}

function EvalSparkline({ series }: { series: PlyEval[] }) {
  const w = 320;
  const h = 100;
  const pad = 8;
  const innerW = w - pad * 2;
  const innerH = h - pad * 2;
  if (series.length < 2) return <p className="muted small">Not enough plies for a chart.</p>;

  const toY = (v: number) => pad + innerH * (0.5 - 0.5 * Math.max(-1, Math.min(1, v)));
  const toX = (i: number) => pad + (innerW * i) / Math.max(1, series.length - 1);

  const pts = series.map((p, i) => `${toX(i)},${toY(p.score)}`).join(" ");

  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h}`} className="eval-sparkline" role="img" aria-label="NN eval by ply">
      <line x1={pad} y1={pad + innerH / 2} x2={w - pad} y2={pad + innerH / 2} stroke="#3d4450" strokeWidth="1" />
      <polyline fill="none" stroke="#7cc4ff" strokeWidth="2" points={pts} />
      {series.map((p, i) => (
        <circle key={p.ply} cx={toX(i)} cy={toY(p.score)} r="2.5" fill="#b9dfff" />
      ))}
    </svg>
  );
}

export default function HomePage() {
  const [game, setGame] = useState(() => new Chess());
  const [userSide, setUserSide] = useState<Side>("w");
  const [status, setStatus] = useState("Choose side and start.");
  const [thinking, setThinking] = useState(false);
  const [score, setScore] = useState<number | null>(null);
  const [topCandidates, setTopCandidates] = useState<string[]>([]);
  const [started, setStarted] = useState(false);
  const inFlightFenRef = useRef<string | null>(null);
  const fenHistoryRef = useRef<string[]>([]);
  const [lastMove, setLastMove] = useState<{ from: string; to: string } | undefined>(undefined);
  const [moveFrom, setMoveFrom] = useState<string | null>(null);
  const [analytics, setAnalytics] = useState<PlyEval[] | null>(null);
  const [analyticsLoading, setAnalyticsLoading] = useState(false);
  const analyzedForFenRef = useRef<string | null>(null);

  const orientation = userSide === "w" ? "white" : "black";
  const userTurn = game.turn() === userSide;
  const gameOver = game.isGameOver();
  const finalKey = gameOver ? game.fen() : "";

  const resultText = useMemo(() => {
    if (!gameOver) return "";
    if (game.isCheckmate()) return `Checkmate — ${game.turn() === "w" ? "Black" : "White"} wins`;
    if (game.isDraw()) return "Draw";
    return "Game over";
  }, [game, gameOver]);

  const moveRows = useMemo(() => {
    const h = game.history();
    const rows: { w?: string; b?: string }[] = [];
    for (let i = 0; i < h.length; i += 2) {
      rows.push({ w: h[i], b: h[i + 1] });
    }
    return rows;
  }, [game]);

  const pgnText = useMemo(() => {
    try {
      return game.pgn();
    } catch {
      return "";
    }
  }, [game]);

  const startGame = (side: Side) => {
    const g = new Chess();
    setUserSide(side);
    setGame(g);
    setScore(null);
    setTopCandidates([]);
    setStarted(true);
    setStatus("Game started.");
    fenHistoryRef.current = [g.fen()];
    setLastMove(undefined);
    setMoveFrom(null);
    setAnalytics(null);
    analyzedForFenRef.current = null;
  };

  const requestNnMove = async (fen: string) => {
    if (inFlightFenRef.current === fen) return;
    inFlightFenRef.current = fen;
    setThinking(true);
    try {
      const res = await fetch("/api/nn-move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen })
      });
      const data = (await res.json()) as MoveResponse & { error?: string };
      if (!res.ok || data.error) {
        throw new Error(data.error || "Failed to get NN move");
      }
      if (!data.move_uci) {
        setStatus("NN has no legal move.");
        return;
      }

      const g2 = new Chess(fen);
      const move = g2.move({
        from: data.move_uci.slice(0, 2),
        to: data.move_uci.slice(2, 4),
        promotion: data.move_uci.length > 4 ? data.move_uci[4] : undefined
      } as Move);

      if (!move) {
        setStatus("NN returned illegal move.");
        return;
      }
      setGame(g2);
      fenHistoryRef.current.push(g2.fen());
      setLastMove({ from: move.from, to: move.to });
      setScore(data.model_score);
      setTopCandidates(data.top_candidates || []);
      setStatus(`NN played ${move.san}`);
    } catch (e) {
      setStatus(`Error: ${String(e)}`);
    } finally {
      inFlightFenRef.current = null;
      setThinking(false);
    }
  };

  useEffect(() => {
    if (!started || gameOver || thinking) return;
    if (!userTurn) {
      requestNnMove(game.fen());
    }
  }, [started, game, userTurn, gameOver, thinking]);

  useEffect(() => {
    if (!gameOver || !started || !finalKey) return;
    if (analyzedForFenRef.current === finalKey) return;
    const fens = fenHistoryRef.current;
    if (fens.length < 2) {
      analyzedForFenRef.current = finalKey;
      setAnalytics([]);
      return;
    }

    analyzedForFenRef.current = finalKey;
    let cancelled = false;
    setAnalyticsLoading(true);
    setAnalytics(null);

    mapLimit(fens, 6, async (fen) => {
      const res = await fetch("/api/nn-eval", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen }),
        cache: "no-store"
      });
      const data = (await res.json()) as { model_score?: number; error?: string };
      if (!res.ok || data.error) {
        throw new Error(data.error || "eval failed");
      }
      return data.model_score ?? 0;
    })
      .then((scores) => {
        if (!cancelled) {
          setAnalytics(scores.map((s, ply) => ({ ply, score: s })));
        }
      })
      .catch(() => {
        if (!cancelled) {
          setAnalytics(null);
          analyzedForFenRef.current = null;
        }
      })
      .finally(() => {
        if (!cancelled) setAnalyticsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [gameOver, started, finalKey]);

  useEffect(() => {
    if (!userTurn) setMoveFrom(null);
  }, [userTurn]);

  const legalFromSelection = useMemo(() => {
    if (!moveFrom || !started || gameOver || !userTurn || thinking) return [];
    try {
      return new Chess(game.fen()).moves({ square: moveFrom as ChessSquare, verbose: true });
    } catch {
      return [];
    }
  }, [moveFrom, game, started, gameOver, userTurn, thinking]);

  const tryUserMove = (from: string, to: string): boolean => {
    if (!started || gameOver || !userTurn || thinking) return false;

    const g2 = new Chess(game.fen());
    const move = g2.move({
      from,
      to,
      promotion: "q"
    });

    if (!move) {
      setStatus("Illegal move.");
      return false;
    }

    setGame(g2);
    fenHistoryRef.current.push(g2.fen());
    setLastMove({ from: move.from, to: move.to });
    setMoveFrom(null);
    setStatus(`You played ${move.san}`);
    return true;
  };

  const onDrop = (sourceSquare: string, targetSquare: string) => {
    return tryUserMove(sourceSquare, targetSquare);
  };

  const handleSquareClick = (square: string, piece: string | undefined) => {
    if (!started || gameOver || !userTurn || thinking) return;

    if (moveFrom) {
      const hit = legalFromSelection.find((m) => m.to === square);
      if (hit && square !== moveFrom) {
        tryUserMove(moveFrom, square);
        return;
      }
    }

    if (piece) {
      const c = piece[0];
      if ((c === "w" || c === "b") && c === game.turn() && c === userSide) {
        setMoveFrom(square === moveFrom ? null : square);
        return;
      }
    }

    setMoveFrom(null);
  };

  const squareStyles = useMemo(() => {
    const styles: Record<string, { backgroundColor?: string; background?: string; boxShadow?: string }> = {};

    if (lastMove) {
      styles[lastMove.from] = { backgroundColor: "rgba(155, 199, 0, 0.41)" };
      styles[lastMove.to] = { backgroundColor: "rgba(155, 199, 0, 0.41)" };
    }

    if (moveFrom && legalFromSelection.length > 0) {
      styles[moveFrom] = {
        ...styles[moveFrom],
        backgroundColor: "rgba(255, 220, 90, 0.55)",
        boxShadow: "inset 0 0 0 2px rgba(200, 160, 0, 0.85)"
      };
      for (const m of legalFromSelection) {
        if (m.to === moveFrom) continue;
        const capture = Boolean(m.captured);
        styles[m.to] = {
          ...styles[m.to],
          ...(capture
            ? {
                backgroundColor: "rgba(255, 120, 55, 0.45)",
                boxShadow: "inset 0 0 0 2px rgba(255, 200, 80, 0.9)"
              }
            : {
                boxShadow: "inset 0 0 12px 4px rgba(40, 140, 255, 0.55)"
              })
        };
      }
    }

    if (game.inCheck() && !gameOver) {
      const ks = kingSquare(game, game.turn());
      if (ks) {
        styles[ks] = {
          ...styles[ks],
          backgroundColor: "rgba(220, 90, 90, 0.5)"
        };
      }
    }

    return styles;
  }, [game, gameOver, lastMove, moveFrom, legalFromSelection]);

  const copyPgn = async () => {
    if (!pgnText) return;
    try {
      await navigator.clipboard.writeText(pgnText);
      setStatus("PGN copied.");
    } catch {
      setStatus("Could not copy PGN.");
    }
  };

  return (
    <main className="lichess-shell">
      <header className="top-bar">
        <div>
          <h1 className="title">Chess2.0</h1>
          <p className="tagline muted">Play vs neural network · Python model server</p>
        </div>
        <div className="header-actions">
          <button className="btn" type="button" onClick={() => startGame("w")} disabled={thinking}>
            White
          </button>
          <button className="btn secondary" type="button" onClick={() => startGame("b")} disabled={thinking}>
            Black
          </button>
        </div>
      </header>

      <div className="game-columns">
        <section className="board-wrap panel">
          <Chessboard
            id="play-vs-nn-board"
            position={game.fen()}
            boardOrientation={orientation}
            onPieceDrop={onDrop}
            onSquareClick={handleSquareClick}
            onPieceClick={(piece, square) => handleSquareClick(square, piece)}
            onPieceDragBegin={(piece, sourceSquare) => {
              if (!started || gameOver || !userTurn || thinking) return;
              const c = piece[0];
              if ((c === "w" || c === "b") && c === game.turn() && c === userSide) {
                setMoveFrom(sourceSquare);
              }
            }}
            onPieceDragEnd={() => setMoveFrom(null)}
            arePiecesDraggable={!thinking && userTurn && !gameOver}
            customDarkSquareStyle={{ backgroundColor: "#b58863" }}
            customLightSquareStyle={{ backgroundColor: "#f0d9b5" }}
            customSquareStyles={squareStyles}
          />
          {game.inCheck() && !gameOver && <div className="check-badge">Check</div>}
        </section>

        <aside className="side">
          <div className="panel side-section">
            <h2 className="panel-title">Game</h2>
            <p className="status-line">{status}</p>
            <dl className="stats">
              <div>
                <dt>Turn</dt>
                <dd>{game.turn() === "w" ? "White" : "Black"}</dd>
              </div>
              <div>
                <dt>Move</dt>
                <dd>{game.moveNumber()}</dd>
              </div>
              <div>
                <dt>NN value</dt>
                <dd>{score === null ? "—" : score.toFixed(3)}</dd>
              </div>
            </dl>
            {thinking && <p className="muted small">Opponent thinking…</p>}
            {resultText && (
              <p className="result-banner">
                <strong>{resultText}</strong>
              </p>
            )}
            <p className="muted small top-cands">
              Top policy: {topCandidates.length ? topCandidates.join(", ") : "—"}
            </p>
          </div>

          <div className="panel side-section movelist-panel">
            <h2 className="panel-title">Moves</h2>
            {moveRows.length === 0 ? (
              <p className="muted small">No moves yet.</p>
            ) : (
              <div className="movelist-grid">
                <div className="movelist-hdr">#</div>
                <div className="movelist-hdr">White</div>
                <div className="movelist-hdr">Black</div>
                {moveRows.map((row, i) => (
                  <Fragment key={i}>
                    <div className="movelist-num">{i + 1}.</div>
                    <div className="movelist-san">{row.w ?? ""}</div>
                    <div className="movelist-san">{row.b ?? ""}</div>
                  </Fragment>
                ))}
              </div>
            )}
            {pgnText ? (
              <button type="button" className="btn secondary full-width" onClick={copyPgn}>
                Copy PGN
              </button>
            ) : null}
          </div>

          <div className="panel side-section analytics-panel">
            <h2 className="panel-title">Post-game analytics</h2>
            <p className="muted small">
              Raw NN value per position (training blend; not engine centipawns). Built from replayed FENs.
            </p>
            {analyticsLoading && <p className="muted small">Computing evals…</p>}
            {!analyticsLoading && analytics && analytics.length >= 2 && <EvalSparkline series={analytics} />}
            {gameOver &&
              !analyticsLoading &&
              analytics === null &&
              fenHistoryRef.current.length >= 2 && (
                <p className="muted small">Eval chart unavailable (request failed).</p>
              )}
            {gameOver && !analyticsLoading && analytics && analytics.length < 2 && (
              <p className="muted small">Not enough positions for a chart.</p>
            )}
            {!gameOver && <p className="muted small">Chart appears when the game ends.</p>}
          </div>
        </aside>
      </div>
    </main>
  );
}
