import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import chess
import chess.engine
import json
import datetime
from tqdm import tqdm
from model.chess_model import ChessModel
from utils.fen_utils import fen_to_tensor
from utils.generate_move_mask import generate_move_mask
from utils.move_index_encoding import policy_index_to_move, move_to_policy_index

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path to Stockfish binary
STOCKFISH_PATH = "/usr/games/stockfish"  # Adjust as needed


# ─────────────────────────────────────────────────────────────────────────────
# Your existing predict function
# ─────────────────────────────────────────────────────────────────────────────

def predict(fen, model):
    board = chess.Board(fen)
    if board.is_checkmate():
        return None, (1.0 if board.turn == chess.BLACK else -1.0), []
    if board.is_stalemate() or board.is_insufficient_material():
        return None, 0.0, []

    board_tensor = torch.tensor(
        fen_to_tensor(fen), dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)
    mask = torch.tensor(
        generate_move_mask(fen), dtype=torch.float32
    ).to(DEVICE)

    if mask.sum().item() == 0:
        return list(board.legal_moves)[0].uci(), 0.0, []

    with torch.no_grad():
        policy_logits, value = model(board_tensor)

    policy_logits = policy_logits.squeeze(0) + (mask - 1) * 1e9
    best_idx      = torch.argmax(policy_logits).item()
    move_uci      = policy_index_to_move(best_idx).uci()

    legal_ucis = [m.uci() for m in board.legal_moves]
    if move_uci not in legal_ucis:
        logits_np  = policy_logits.cpu().numpy()
        best_score = -float('inf')
        move_uci   = legal_ucis[0]
        for lm in board.legal_moves:
            idx = move_to_policy_index(lm)
            if idx is not None and logits_np[idx] > best_score:
                best_score = logits_np[idx]
                move_uci   = lm.uci()

    # top-5 legal moves
    top5_indices = torch.topk(
        policy_logits, min(10, policy_logits.shape[0])
    ).indices.tolist()
    top5 = []
    for idx in top5_indices:
        try:
            m = policy_index_to_move(idx).uci()
            if m in legal_ucis and m not in top5:
                top5.append(m)
            if len(top5) >= 5:
                break
        except:
            pass

    score = float(torch.clamp(value, -1.0, 1.0).item())
    return move_uci, score, top5


# ─────────────────────────────────────────────────────────────────────────────
# Test positions
# ─────────────────────────────────────────────────────────────────────────────

COMPARISON_SUITE = [
    # Openings
    {
        "name": "Starting position — White",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "category": "Opening"
    },
    {
        "name": "After 1.e4 — Black response",
        "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "category": "Opening"
    },
    {
        "name": "Italian Game — White develops",
        "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3",
        "category": "Opening"
    },
    {
        "name": "Sicilian Defense",
        "fen": "rnbqkb1r/pp1ppppp/5n2/2p5/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
        "category": "Opening"
    },
    
    # Tactics
    {
        "name": "Scholar's mate Qxf7#",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "category": "Tactic"
    },
    {
        "name": "Knight fork on d5",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 1",
        "category": "Tactic"
    },
    {
        "name": "Bishop sacrifice Bxf7+",
        "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 1",
        "category": "Tactic"
    },
    {
        "name": "Discovered check with Ng5",
        "fen": "r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        "category": "Tactic"
    },
    {
        "name": "Queen fork",
        "fen": "r2qkb1r/ppp2ppp/2np1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 1",
        "category": "Tactic"
    },
    
    # Endgames
    {
        "name": "Pawn promotion a8=Q",
        "fen": "8/P7/8/8/8/8/8/4K1k1 w - - 0 1",
        "category": "Endgame"
    },
    {
        "name": "K+R vs K — rook checkmate",
        "fen": "8/8/8/8/8/3k4/8/3RK3 w - - 0 1",
        "category": "Endgame"
    },
    {
        "name": "K+Q vs K — avoid stalemate",
        "fen": "7k/8/6QK/8/8/8/8/8 w - - 0 1",
        "category": "Endgame"
    },
    {
        "name": "K+P vs K — pawn endgame",
        "fen": "8/8/8/8/8/3k4/3p4/3K4 b - - 0 1",
        "category": "Endgame"
    },
    {
        "name": "King opposition",
        "fen": "8/8/8/3k4/8/3K4/8/8 w - - 0 1",
        "category": "Endgame"
    },
    
    # Mating patterns
    {
        "name": "Back rank mate threat",
        "fen": "6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1",
        "category": "Mating Pattern"
    },
    {
        "name": "Smothered mate setup",
        "fen": "7k/6pp/8/8/8/8/8/Q5K1 w - - 0 1",
        "category": "Mating Pattern"
    },
    {
        "name": "Two rooks mate",
        "fen": "k7/8/1KR5/2R5/8/8/8/8 w - - 0 1",
        "category": "Mating Pattern"
    },
    {
        "name": "King and queen vs king",
        "fen": "8/8/8/8/8/2k5/8/2KQ4 w - - 0 1",
        "category": "Mating Pattern"
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Stockfish evaluation
# ─────────────────────────────────────────────────────────────────────────────

def get_stockfish_eval_local(fen, depth=15):
    """Get Stockfish evaluation using local engine"""
    try:
        # Try to find stockfish
        import shutil
        stockfish_path = shutil.which("stockfish")
        if not stockfish_path and os.path.exists(STOCKFISH_PATH):
            stockfish_path = STOCKFISH_PATH
        
        if not stockfish_path:
            return None
        
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            board = chess.Board(fen)
            analysis = engine.analyse(
                board, 
                chess.engine.Limit(depth=depth),
                multipv=3
            )
            
            results = []
            for line in analysis:
                move = line['pv'][0].uci() if line['pv'] else None
                score = line['score']
                
                if score.relative.is_mate():
                    cp = None
                    mate = score.relative.mate()
                else:
                    cp = score.relative.score()
                    mate = None
                
                results.append({
                    'move': move,
                    'cp': cp,
                    'mate': mate,
                    'pv': [m.uci() for m in line['pv'][:5]]
                })
            return results
    except Exception as e:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Comparison function
# ─────────────────────────────────────────────────────────────────────────────

def compare_with_stockfish(model, test_positions):
    """Compare model predictions with Stockfish"""
    
    results = []
    
    for test in tqdm(test_positions, desc="Comparing", unit="pos"):
        fen = test["fen"]
        name = test["name"]
        category = test["category"]
        
        # Get model prediction
        try:
            model_move, model_score, model_top5 = predict(fen, model)
        except Exception as e:
            print(f"Error in model prediction for {name}: {e}")
            model_move, model_score, model_top5 = None, 0.0, []
        
        # Get Stockfish evaluation
        sf_evals = get_stockfish_eval_local(fen, depth=16)
        
        if sf_evals and len(sf_evals) > 0:
            sf_best = sf_evals[0]
            sf_move = sf_best['move']
            
            if sf_best['mate']:
                sf_score_str = f"mate in {sf_best['mate']}"
            else:
                sf_score_str = f"{sf_best['cp']}cp"
            
            match_exact = (model_move == sf_move) if model_move else False
            match_top3 = any(model_move == ev['move'] for ev in sf_evals[:3]) if model_move else False
            
            # Check if model's top-3 includes any Stockfish top-3
            top3_overlap = any(m in model_top5[:3] for m in [ev['move'] for ev in sf_evals[:3]]) if model_top5 else False
            
        else:
            sf_move = "N/A"
            sf_score_str = "?"
            match_exact = False
            match_top3 = False
            top3_overlap = False
        
        results.append({
            "name": name,
            "category": category,
            "fen": fen,
            "model_move": model_move,
            "model_score": model_score,
            "model_top5": model_top5[:3] if model_top5 else [],
            "stockfish_move": sf_move,
            "stockfish_score": sf_score_str,
            "match_exact": match_exact,
            "match_top3": match_top3,
            "top3_overlap": top3_overlap
        })
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Print report (FIXED for None values)
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_report(results):
    """Print detailed comparison results with proper None handling"""
    
    print("\n" + "=" * 90)
    print("MODEL vs STOCKFISH COMPARISON REPORT")
    print("=" * 90)
    
    # Per-position results
    for r in results:
        print(f"\n📌 {r['name']} ({r['category']})")
        
        # Handle None model_move
        model_move_str = r['model_move'] if r['model_move'] else "None (checkmate/stalemate)"
        print(f"   🤖 Model:    {model_move_str:>12} (score={r['model_score']:.3f})")
        
        if r['model_top5']:
            print(f"      Top-3:    {', '.join(r['model_top5'])}")
        
        print(f"   🎯 Stockfish: {r['stockfish_move']:>12} ({r['stockfish_score']})")
        
        if r['match_exact']:
            print(f"   ✅ EXACT MATCH!")
        elif r['match_top3']:
            print(f"   👍 In Stockfish's top-3")
        elif r['top3_overlap']:
            print(f"   📍 Top-3 overlap (model's top-3 includes Stockfish move)")
        else:
            print(f"   ❌ Different recommendation")
    
    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    
    total = len(results)
    exact = sum(1 for r in results if r['match_exact'])
    top3 = sum(1 for r in results if r['match_top3'])
    overlap = sum(1 for r in results if r['top3_overlap'])
    
    print(f"\nTotal positions: {total}")
    print(f"✅ Exact matches:  {exact}/{total} ({exact/total*100:.1f}%)")
    print(f"👍 In top-3:       {top3}/{total} ({top3/total*100:.1f}%)")
    print(f"📍 Top-3 overlap:  {overlap}/{total} ({overlap/total*100:.1f}%)")
    
    # By category
    print("\n📂 By Category:")
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = {'exact': 0, 'total': 0}
        categories[cat]['total'] += 1
        if r['match_exact']:
            categories[cat]['exact'] += 1
    
    for cat, stats in categories.items():
        pct = stats['exact'] / stats['total'] * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"   {cat:<15} {bar} {stats['exact']}/{stats['total']} ({pct:.0f}%)")
    
    # Performance analysis
    print("\n📊 Performance Analysis:")
    print(f"   Openings:     {sum(1 for r in results if r['category']=='Opening' and r['match_exact'])}/4 exact matches")
    print(f"   Tactics:      {sum(1 for r in results if r['category']=='Tactic' and r['match_exact'])}/5 exact matches")
    print(f"   Endgames:     {sum(1 for r in results if r['category']=='Endgame' and r['match_exact'])}/5 exact matches")
    print(f"   Mating:       {sum(1 for r in results if r['category']=='Mating Pattern' and r['match_exact'])}/4 exact matches")
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("outputs/comparisons", exist_ok=True)
    output_file = f"outputs/comparisons/comparison_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to {output_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Verify model file
    model_path = "outputs/models/latest.pt"
    print("\n" + "="*60)
    print("MODEL VERIFICATION")
    print("="*60)
    
    if not os.path.exists(model_path):
        print(f"❌ ERROR: {model_path} not found!")
        if os.path.exists("outputs/models"):
            print("Available checkpoints:")
            for f in os.listdir("outputs/models"):
                if f.endswith('.pt'):
                    print(f"   - {f}")
        exit(1)
    
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(model_path))
    print(f"📁 Loading: {os.path.abspath(model_path)}")
    print(f"📊 Size: {file_size:.2f} MB")
    print(f"🕐 Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load model
    print("\n🔄 Loading model...")
    model, _, _, shard_id = ChessModel.load_checkpoint(model_path, device=DEVICE)
    model.eval()
    print(f"✅ Model loaded — shard {shard_id}")
    print(f"📱 Device: {DEVICE}")
    
    # Check Stockfish
    import shutil
    if shutil.which("stockfish") or os.path.exists(STOCKFISH_PATH):
        print(f"✅ Stockfish found")
    else:
        print(f"⚠️  Stockfish not found. Install with: sudo apt-get install stockfish")
    
    # Run comparison
    print(f"\n🔄 Comparing on {len(COMPARISON_SUITE)} positions...")
    results = compare_with_stockfish(model, COMPARISON_SUITE)
    
    # Print report
    print_comparison_report(results)