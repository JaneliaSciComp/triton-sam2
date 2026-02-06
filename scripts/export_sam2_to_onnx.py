#!/usr/bin/env python3
"""
Export SAM2 model to ONNX format for NVIDIA Triton deployment.

This script exports the SAM2 encoder and decoder separately:
- Encoder: Processes the image and generates embeddings (expensive operation)
- Decoder: Takes embeddings and prompts to generate segmentation masks (fast operation)
"""

import argparse
import warnings
from pathlib import Path

import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def export_encoder(model, output_path, image_size=1024):
    """Export the SAM2 image encoder to ONNX format."""
    print(f"Exporting encoder to {output_path}...")

    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    # Get the encoder model
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
    """Export the SAM2 mask decoder to ONNX format."""
    print(f"Exporting decoder to {output_path}...")

    # Get embedding dimensions from the model
    embed_dim = model.sam_prompt_encoder.embed_dim
    embed_size = model.sam_prompt_encoder.image_embedding_size

    # Create dummy inputs for the decoder
    # These match the expected inputs during inference
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float32),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float32),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float32),
    }

    # Create a wrapper class for the decoder
    class SAM2DecoderONNX(torch.nn.Module):
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
            # For SAM2.1, set use_high_res_features to False by temporarily disabling it
            use_high_res = self.sam_mask_decoder.use_high_res_features
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
            self.sam_mask_decoder.use_high_res_features = use_high_res

            return low_res_masks, iou_predictions

    decoder_onnx = SAM2DecoderONNX(model)
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
    parser = argparse.ArgumentParser(description="Export SAM2 to ONNX for Triton")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to SAM2 checkpoint file"
    )
    parser.add_argument(
        "--model-cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="SAM2 model config name (relative to sam2 package, e.g., configs/sam2.1/sam2.1_hiera_b+.yaml)"
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
    encoder_dir = output_dir / "sam2_encoder" / "1"
    decoder_dir = output_dir / "sam2_decoder" / "1"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    decoder_dir.mkdir(parents=True, exist_ok=True)

    # Build SAM2 model
    print(f"Loading SAM2 model from {args.checkpoint}...")
    model = build_sam2(args.model_cfg, args.checkpoint, device=args.device)
    model.eval()
    print("✓ Model loaded successfully")

    # Export encoder
    encoder_path = encoder_dir / "model.onnx"
    export_encoder(model, encoder_path, args.image_size)

    # Export decoder
    decoder_path = decoder_dir / "model.onnx"
    export_decoder(model, decoder_path)

    print("\n✓ Export complete!")
    print(f"  Encoder: {encoder_path}")
    print(f"  Decoder: {decoder_path}")


if __name__ == "__main__":
    main()
