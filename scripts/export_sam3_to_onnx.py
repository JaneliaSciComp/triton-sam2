#!/usr/bin/env python3
"""
Export SAM3 Tracker model to ONNX format for NVIDIA Triton deployment.

This script exports the SAM3 Tracker encoder and decoder separately:
- Encoder: Processes the image and generates embeddings (expensive operation)
- Decoder: Takes embeddings and prompts to generate segmentation masks (fast operation)

SAM3 Tracker is backward compatible with SAM2's architecture for visual prompts.
"""

import argparse
import warnings
from pathlib import Path

import torch
import numpy as np


def export_encoder(model, output_path, image_size=1024):
    """Export the SAM3 Tracker image encoder to ONNX format."""
    print(f"Exporting SAM3 Tracker encoder to {output_path}...")

    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    # Get the encoder model
    # SAM3 Tracker should have an image_encoder similar to SAM2
    encoder = model.image_encoder
    encoder.eval()

    # Export to ONNX
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        torch.onnx.export(
            encoder,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["image_embeddings"],
            dynamic_axes={
                "image": {0: "batch_size"},
                "image_embeddings": {0: "batch_size"}
            }
        )

    print(f"✓ Encoder exported successfully")


def export_decoder(model, output_path):
    """Export the SAM3 Tracker mask decoder to ONNX format."""
    print(f"Exporting SAM3 Tracker decoder to {output_path}...")

    # Get embedding dimensions from the model
    # Try to access the same attributes as SAM2
    try:
        embed_dim = model.sam_prompt_encoder.embed_dim
        embed_size = model.sam_prompt_encoder.image_embedding_size
    except AttributeError:
        # Fallback to common SAM dimensions
        print("Warning: Could not auto-detect embedding dimensions, using defaults")
        embed_dim = 256
        embed_size = (64, 64)

    # Create dummy inputs for the decoder
    # These match the expected inputs during inference
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float32),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float32),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float32),
    }

    # Create a wrapper class for the decoder
    class SAM3DecoderONNX(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.image_encoder = model.image_encoder
            self.sam_prompt_encoder = model.sam_prompt_encoder
            self.sam_mask_decoder = model.sam_mask_decoder

        def forward(self, image_embeddings, point_coords, point_labels):
            # Encode prompts
            sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )

            # Predict masks
            # Disable high_res_features if present (SAM2.1 feature)
            use_high_res = getattr(self.sam_mask_decoder, 'use_high_res_features', False)
            if hasattr(self.sam_mask_decoder, 'use_high_res_features'):
                self.sam_mask_decoder.use_high_res_features = False

            low_res_masks, iou_predictions, _, _ = self.sam_mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=None,
            )

            # Restore original setting
            if hasattr(self.sam_mask_decoder, 'use_high_res_features'):
                self.sam_mask_decoder.use_high_res_features = use_high_res

            return low_res_masks, iou_predictions

    decoder_onnx = SAM3DecoderONNX(model)
    decoder_onnx.eval()

    # Export to ONNX
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        torch.onnx.export(
            decoder_onnx,
            tuple(dummy_inputs.values()),
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=["masks", "iou_predictions"],
            dynamic_axes={
                "image_embeddings": {0: "batch_size"},
                "point_coords": {0: "batch_size", 1: "num_points"},
                "point_labels": {0: "batch_size", 1: "num_points"},
                "masks": {0: "batch_size"},
                "iou_predictions": {0: "batch_size"}
            }
        )

    print(f"✓ Decoder exported successfully")


def main():
    parser = argparse.ArgumentParser(description="Export SAM3 Tracker to ONNX for Triton")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to SAM3 checkpoint file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model_repository",
        help="Output directory for ONNX models"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Input image size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for export (cpu or cuda)"
    )

    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    encoder_dir = output_dir / "sam3_encoder" / "1"
    decoder_dir = output_dir / "sam3_decoder" / "1"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    decoder_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SAM3 Tracker ONNX Export")
    print("=" * 70)
    print()

    # Try different SAM3 loading methods
    print(f"Loading SAM3 Tracker model from {args.checkpoint}...")

    try:
        # Method 1: Try using sam3 package if installed
        try:
            from sam3.build_sam import build_sam3
            from sam3.sam3_tracker import Sam3Tracker

            print("Using sam3 package...")
            # Load SAM3 model
            model = build_sam3(checkpoint=args.checkpoint, device=args.device)

            # Get the tracker component
            if hasattr(model, 'tracker'):
                model = model.tracker

        except ImportError:
            # Method 2: Try using sam2 API if sam3 is compatible
            from sam2.build_sam import build_sam2

            print("SAM3 package not found, attempting to load as SAM2-compatible model...")
            model = build_sam2(
                config_file=None,  # SAM3 may not need config
                ckpt_path=args.checkpoint,
                device=args.device
            )

        model.eval()
        print("✓ Model loaded successfully")

    except Exception as e:
        print(f"Error loading model: {e}")
        print()
        print("Make sure you have installed SAM3:")
        print("  pixi run install-sam3")
        print()
        print("Or that the checkpoint path is correct.")
        raise

    # Export encoder
    encoder_path = encoder_dir / "model.onnx"
    export_encoder(model, encoder_path, args.image_size)

    # Export decoder
    decoder_path = decoder_dir / "model.onnx"
    export_decoder(model, decoder_path)

    print()
    print("=" * 70)
    print("✓ Export complete!")
    print("=" * 70)
    print(f"  Encoder: {encoder_path}")
    print(f"  Decoder: {decoder_path}")
    print()
    print("Next steps:")
    print("  1. Create Triton config files:")
    print("     - model_repository/sam3_encoder/config.pbtxt")
    print("     - model_repository/sam3_decoder/config.pbtxt")
    print("  2. Start/restart Triton server: docker compose restart")
    print("  3. Test with client: pixi run test-sam3")


if __name__ == "__main__":
    main()
