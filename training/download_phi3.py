# AI-Generated | Claude (Anthropic) | AgentZero | 2026-03-11
"""
Downloads Phi-3 Mini from HuggingFace to local storage.
Run this once. It downloads ~4.5GB.

Usage: python training/download_phi3.py
"""

import os
import sys

MODEL_DIR = "Z:/AgentZero/models/phi3-mini"
REPO_ID = "microsoft/Phi-3-mini-4k-instruct"

def download():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Check if already downloaded
    existing = os.listdir(MODEL_DIR)
    if any('config.json' in f for f in existing):
        safetensors = [f for f in existing if f.endswith('.safetensors')]
        total = sum(os.path.getsize(os.path.join(MODEL_DIR, f)) for f in existing if os.path.isfile(os.path.join(MODEL_DIR, f)))
        print(f"Model appears already downloaded: {len(existing)} files, {total/1e9:.2f}GB")
        print(f"Safetensors shards: {len(safetensors)}")
        return

    try:
        from huggingface_hub import snapshot_download
        print(f"Downloading {REPO_ID}...")
        print(f"Destination: {MODEL_DIR}")
        print("This is ~4.5GB and will take several minutes.")
        print("Do not interrupt.")
        print()

        path = snapshot_download(
            repo_id=REPO_ID,
            local_dir=MODEL_DIR,
            ignore_patterns=["*.gguf", "*.bin", "original/*"],  # safetensors only
        )
        print(f"\nDownload complete: {path}")
        files = os.listdir(MODEL_DIR)
        total = sum(os.path.getsize(os.path.join(MODEL_DIR, f)) for f in files if os.path.isfile(os.path.join(MODEL_DIR, f)))
        print(f"Total: {len(files)} files, {total/1e9:.2f}GB")

    except ImportError:
        print("huggingface_hub not installed. Installing...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        print("Retry: python training/download_phi3.py")

    except Exception as e:
        print(f"Download error: {e}")
        print("If this is a 401 error, run: huggingface-cli login")
        print("If this is a network error, check internet connection.")
        print(f"Partial download in: {MODEL_DIR}")
        raise

if __name__ == "__main__":
    download()
