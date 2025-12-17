#!/usr/bin/env python3
"""
Download pre-exported SAM3 Tracker ONNX models from onnx-community

Downloads the vision encoder and prompt encoder + mask decoder models.
"""

import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

# Configuration
MODEL_REPO = "onnx-community/sam3-tracker-ONNX"
OUTPUT_DIR = Path("model_repository")

# Models to download (using FP32 versions for best quality)
MODELS = {
    "sam3_encoder": [
        "onnx/vision_encoder.onnx",
        "onnx/vision_encoder.onnx_data",
    ],
    "sam3_decoder": [
        "onnx/prompt_encoder_mask_decoder.onnx",
        "onnx/prompt_encoder_mask_decoder.onnx_data",
    ],
}


def download_model(model_name, files):
    """Download ONNX model files for a specific model."""
    model_dir = OUTPUT_DIR / model_name / "1"
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Downloading {model_name}")
    print(f"{'='*70}")

    for file in files:
        print(f"\n  Downloading {file}...")
        output_name = "model.onnx" if file.endswith(".onnx") and not file.endswith("_data") else "model.onnx_data"
        output_path = model_dir / output_name

        try:
            downloaded_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=file,
                local_dir=str(OUTPUT_DIR / "temp"),
                local_dir_use_symlinks=False,
            )

            # Move to correct location with standard naming
            import shutil
            shutil.move(downloaded_path, output_path)
            print(f"  ✓ Saved to: {output_path}")

        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            return False

    return True


def main():
    print("=" * 70)
    print("Downloading SAM3 Tracker ONNX Models")
    print("=" * 70)
    print(f"\nSource: {MODEL_REPO}")
    print(f"Target: {OUTPUT_DIR}/")
    print()

    success_count = 0
    for model_name, files in MODELS.items():
        if download_model(model_name, files):
            success_count += 1

    print("\n" + "=" * 70)
    if success_count == len(MODELS):
        print("✓ All models downloaded successfully!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Restart Triton: docker compose restart")
        print("  2. Run tests: pixi run test-sam3")
        print()
        return 0
    else:
        print(f"✗ Downloaded {success_count}/{len(MODELS)} models")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
