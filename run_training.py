# run_training.py - Simple entry point for unified training
"""Run the unified training pipeline - combines regular + endgame data"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"📋 {description}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ Error in {description}")
        return False
    return True

def main():
    """Run the complete training pipeline"""
    
    print("\n" + "="*60)
    print("UNIFIED TRAINING PIPELINE")
    print("="*60)
    
    # Check if endgame dataset exists
    if not os.path.exists("data/endgame_shards"):
        print("⚠️ Endgame dataset not found!")
        print("Please run: python preprocessing/generate_endgame.py first")
        response = input("Generate endgame dataset now? (y/n): ")
        if response.lower() == 'y':
            if not run_command(
                "python preprocessing/generate_endgame.py",
                "Generating Endgame Dataset"
            ):
                print("Failed to generate endgame dataset. Exiting...")
                return
        else:
            print("Cannot proceed without endgame data. Exiting...")
            return
    
    # Check if regular model exists
    if not os.path.exists("outputs/models/latest.pt"):
        print("⚠️ No existing model found at outputs/models/latest.pt")
        print("Please train a regular model first or provide a model file")
        return
    
    print("\n✅ Found regular model: outputs/models/latest.pt")
    print("✅ Found endgame shards in: data/endgame_shards")
    print("\n🚀 Starting unified training to combine both datasets...")
    
    # Run unified training (this will update latest.pt with endgame knowledge)
    if not run_command(
        "python training/train.py",
        "Running Unified Training"
    ):
        print("Training failed!")
        return
    
    print("\n" + "="*60)
    print("✅ UNIFIED TRAINING COMPLETE!")
    print("✅ Your model now knows regular positions AND endgames!")
    print(f"✅ Model saved to: outputs/models/latest.pt")
    print("="*60)

if __name__ == "__main__":
    main()