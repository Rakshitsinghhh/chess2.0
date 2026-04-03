/**
 * Move indexing matching utils/move_index_encoding.py (73 logits per from-square).
 */

export const TOTAL_MOVES = 4672;

const DIRS: [number, number][] = [
  [0, 1],
  [1, 0],
  [0, -1],
  [-1, 0],
  [1, 1],
  [1, -1],
  [-1, -1],
  [-1, 1]
];

const KNIGHT_DIRS: [number, number][] = [
  [1, 2],
  [2, 1],
  [2, -1],
  [1, -2],
  [-1, -2],
  [-2, -1],
  [-2, 1],
  [-1, 2]
];

const PROMO_PIECES = ["q", "r", "b", "n"] as const;

export function sqToCoords(sq: string): [number, number] {
  const file = sq.charCodeAt(0) - "a".charCodeAt(0);
  const rank = parseInt(sq[1], 10) - 1;
  return [file, rank];
}

export function sqToIndex(sq: string): number {
  const [file, rank] = sqToCoords(sq);
  return rank * 8 + file;
}

export function coordsToSq(x: number, y: number): string {
  if (x < 0 || x > 7 || y < 0 || y > 7) throw new Error("coords out of board");
  return String.fromCharCode("a".charCodeAt(0) + x) + (y + 1);
}

export function moveToPolicyIndex(from: string, to: string, promotion?: string): number {
  const fromIdx = sqToIndex(from);
  const [fx, fy] = sqToCoords(from);
  const [tx, ty] = sqToCoords(to);
  const dx = tx - fx;
  const dy = ty - fy;
  const base = fromIdx * 73;

  if (promotion) {
    const direction = dx + 1;
    const promoIdx = PROMO_PIECES.indexOf(promotion as (typeof PROMO_PIECES)[number]);
    if (promoIdx < 0) throw new Error("Unsupported promotion");
    return base + 64 + direction * 4 + promoIdx;
  }

  const knightIdx = KNIGHT_DIRS.findIndex((d) => d[0] === dx && d[1] === dy);
  if (knightIdx >= 0) {
    return base + 56 + knightIdx;
  }

  for (let dIdx = 0; dIdx < DIRS.length; dIdx++) {
    const [ddx, ddy] = DIRS[dIdx];
    for (let dist = 1; dist < 8; dist++) {
      if (dx === ddx * dist && dy === ddy * dist) {
        return base + dIdx * 7 + (dist - 1);
      }
    }
  }

  throw new Error(`Illegal move for encoding: ${from}${to}`);
}

export function policyIndexToUci(index: number): string {
  const fromSq = Math.floor(index / 73);
  const offset = index % 73;
  const fx = fromSq % 8;
  const fy = Math.floor(fromSq / 8);

  if (offset < 56) {
    const d = Math.floor(offset / 7);
    const dist = (offset % 7) + 1;
    const [ddx, ddy] = DIRS[d];
    const tx = fx + ddx * dist;
    const ty = fy + ddy * dist;
    return coordsToSq(fx, fy) + coordsToSq(tx, ty);
  }

  if (offset < 64) {
    const ki = offset - 56;
    const [ddx, ddy] = KNIGHT_DIRS[ki];
    const tx = fx + ddx;
    const ty = fy + ddy;
    return coordsToSq(fx, fy) + coordsToSq(tx, ty);
  }

  const promo = offset - 64;
  const direction = Math.floor(promo / 4);
  const pieceIdx = promo % 4;
  const piece = PROMO_PIECES[pieceIdx];
  const ddx = direction - 1;
  const ty = fy === 6 ? fy + 1 : fy - 1;
  const tx = fx + ddx;
  return coordsToSq(fx, fy) + coordsToSq(tx, ty) + piece;
}
