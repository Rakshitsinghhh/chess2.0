#!/usr/bin/env python3
"""
Upload chess model weights to HuggingFace Hub.

Usage:
    pip install huggingface_hub
    huggingface-cli login          # or set HF_TOKEN env var
    python scripts/upload_model_hf.py

After uploading, copy the printed MODEL_URL into:
  - nextjs_host/model_server/.env  (for Docker)
  - Railway environment variables  (MODEL_URL)
"""
import os
import sys
import argparse

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    sys.exit("Run: pip install huggingface_hub")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_MODEL_PATH = os.path.join(REPO_ROOT, "outputs", "models", "latest.pt")
DEFAULT_REPO_ID = "chess2.0-model"          # will be prefixed with your HF username


def main():
    parser = argparse.ArgumentParser(description="Upload chess model to HuggingFace Hub")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to .pt file")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF repo name (no username prefix)")
    parser.add_argument("--private", action="store_true", default=True, help="Make repo private")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        sys.exit(f"Model not found: {args.model_path}")

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    # Get the current user
    user_info = api.whoami()
    username = user_info["name"]
    full_repo_id = f"{username}/{args.repo_id}"

    print(f"Creating/verifying repo: {full_repo_id}  (private={args.private})")
    create_repo(
        repo_id=full_repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
        token=token,
    )

    model_size_mb = os.path.getsize(args.model_path) / 1024 / 1024
    print(f"Uploading {args.model_path}  ({model_size_mb:.0f} MB) …")
    api.upload_file(
        path_or_fileobj=args.model_path,
        path_in_repo="latest.pt",
        repo_id=full_repo_id,
        repo_type="model",
        token=token,
    )

    url = f"https://huggingface.co/{full_repo_id}/resolve/main/latest.pt"
    print("\n✅ Upload complete!")
    print(f"\nSet this in Railway / Docker env vars:")
    print(f"  MODEL_URL={url}")
    print(f"\nOr for private repos, add your HF token:")
    print(f"  HF_TOKEN=hf_xxxx")


if __name__ == "__main__":
    main()
