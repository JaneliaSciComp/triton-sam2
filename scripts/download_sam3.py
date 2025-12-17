#!/usr/bin/env python3
"""
Download SAM3 model checkpoints from HuggingFace

Requires HuggingFace authentication and access to facebook/sam3 model.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, whoami
from huggingface_hub.utils import HfHubHTTPError

# Configuration
CHECKPOINT_DIR = Path("checkpoints")
MODEL_REPO = "facebook/sam3"
MODEL_FILE = "sam3.pt"

def main():
    # Create checkpoint directory
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CHECKPOINT_DIR / MODEL_FILE

    # Check if already downloaded
    if output_path.exists():
        print(f"✓ Checkpoint already exists: {output_path}")
        return 0

    print("=" * 70)
    print("Downloading SAM3 Checkpoint from HuggingFace")
    print("=" * 70)
    print()

    # Check authentication
    try:
        user_info = whoami()
        print(f"✓ Logged in as: {user_info['name']}")
        print()
    except Exception as e:
        print("✗ Not logged in to HuggingFace!")
        print()
        print("Please run: pixi run hf-login")
        print()
        print("You'll also need:")
        print("  1. Request access to SAM3: https://huggingface.co/facebook/sam3")
        print("  2. Wait for approval (usually instant)")
        print()
        return 1

    # Download checkpoint
    print(f"Downloading {MODEL_FILE} from {MODEL_REPO}...")
    print("This may take a few minutes (~350MB)...")
    print()

    try:
        downloaded_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=str(CHECKPOINT_DIR),
            local_dir_use_symlinks=False
        )

        print()
        print("=" * 70)
        print("✓ Download Complete!")
        print("=" * 70)
        print(f"Checkpoint saved to: {output_path}")
        print()
        print("Next steps:")
        print("  pixi run setup-sam3")
        print()
        return 0

    except HfHubHTTPError as e:
        if "401" in str(e) or "403" in str(e):
            print()
            print("✗ Access denied!")
            print()
            print("Make sure you have:")
            print("  1. Logged in: pixi run hf-login")
            print("  2. Requested access: https://huggingface.co/facebook/sam3")
            print("  3. Been granted access (check your email)")
            print()
        else:
            print(f"✗ Download failed: {e}")
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
