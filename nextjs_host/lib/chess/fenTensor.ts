/**
 * FEN → (20, 8, 8) float tensor, NCHW layout matching utils/fen_utils.py
 */

const PIECE_TO_CHANNEL: Record<string, number> = {
  P: 0,
  N: 1,
  B: 2,
  R: 3,
  Q: 4,
  K: 5,
  p: 6,
  n: 7,
  b: 8,
  r: 9,
  q: 10,
  k: 11
};

function parseFen(fen: string): { boardPart: string; turn: string; castling: string } {
  const parts = fen.split(" ");
  return { boardPart: parts[0], turn: parts[1] ?? "w", castling: parts[2] ?? "-" };
}

function boardToMatrix(boardPart: string): string[][] {
  const board: string[][] = [];
  const rows = boardPart.split("/");
  for (const row of rows) {
    const currentRow: string[] = [];
    for (const ch of row) {
      if (/\d/.test(ch)) {
        currentRow.push(...Array(parseInt(ch, 10)).fill("."));
      } else {
        currentRow.push(ch);
      }
    }
    board.push(currentRow);
  }
  return board;
}

/** Flat float32 [1, 20, 8, 8] row-major (batch, channels, rows, cols) */
export function fenToInputTensorData(fen: string): Float32Array {
  const { boardPart, turn, castling } = parseFen(fen);
  const board = boardToMatrix(boardPart);
  const data = new Float32Array(1 * 20 * 8 * 8);

  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 8; j++) {
      const piece = board[i][j];
      if (piece !== ".") {
        const channel = PIECE_TO_CHANNEL[piece];
        if (channel !== undefined) {
          data[channel * 64 + i * 8 + j] = 1;
        }
      }
    }
  }

  const turnCh = turn === "w" ? 12 : 13;
  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 8; j++) {
      data[turnCh * 64 + i * 8 + j] = 1;
    }
  }

  if (castling.includes("K")) {
    for (let i = 0; i < 8; i++) {
      for (let j = 0; j < 8; j++) {
        data[14 * 64 + i * 8 + j] = 1;
      }
    }
  }
  if (castling.includes("Q")) {
    for (let i = 0; i < 8; i++) {
      for (let j = 0; j < 8; j++) {
        data[15 * 64 + i * 8 + j] = 1;
      }
    }
  }
  if (castling.includes("k")) {
    for (let i = 0; i < 8; i++) {
      for (let j = 0; j < 8; j++) {
        data[16 * 64 + i * 8 + j] = 1;
      }
    }
  }
  if (castling.includes("q")) {
    for (let i = 0; i < 8; i++) {
      for (let j = 0; j < 8; j++) {
        data[17 * 64 + i * 8 + j] = 1;
      }
    }
  }

  return data;
}
