#!/usr/bin/env python3
"""
Export SAM1 model to ONNX format for NVIDIA Triton deployment.

This script exports the SAM1 (Segment Anything Model v1) encoder and decoder separately:
- Encoder: Processes the image and generates embeddings (expensive operation)
- Decoder: Takes embeddings and prompts to generate segmentation masks (fast operation)

SAM1 model variants:
- vit_h: ViT-Huge (default, 636M params)
- vit_l: ViT-Large (308M params)
- vit_b: ViT-Base (91M params)
"""

import argparse
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

# Add segment-anything to path - check multiple possible locations
SCRIPT_DIR = Path(__file__).parent.absolute()
SAM1_REPO_LOCATIONS = [
    SCRIPT_DIR.parent / "sam1_repo",  # Cloned by pixi task
    SCRIPT_DIR.parent.parent / "segment-anything",  # Sibling directory
]

for repo_path in SAM1_REPO_LOCATIONS:
    if repo_path.exists():
        sys.path.insert(0, str(repo_path))
        break
else:
    print("Warning: segment-anything repository not found. Trying installed package...")

from segment_anything import sam_model_registry


class SAM1EncoderONNX(nn.Module):
    """Wrapper for SAM1 image encoder for ONNX export."""

    def __init__(self, sam_model):
        super().__init__()
        self.image_encoder = sam_model.image_encoder

    def forward(self, image):
        """
        Encode image to embeddings.

        Args:
            image: Preprocessed image tensor (B, 3, 1024, 1024)

        Returns:
            image_embeddings: (B, 256, 64, 64)
        """
        return self.image_encoder(image)


class SAM1DecoderONNX(nn.Module):
    """
    Wrapper for SAM1 decoder for ONNX export.

    This combines the prompt encoder and mask decoder, similar to the
    SamOnnxModel in segment_anything.utils.onnx but optimized for
    Triton deployment with a simpler interface.
    """

    def __init__(self, sam_model, return_single_mask=True):
        super().__init__()
        self.model = sam_model
        self.mask_decoder = sam_model.mask_decoder
        self.prompt_encoder = sam_model.prompt_encoder
        self.img_size = sam_model.image_encoder.img_size
        self.return_single_mask = return_single_mask

    def _embed_points(self, point_coords, point_labels):
        """Embed point prompts."""
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.prompt_encoder.not_a_point_embed.weight * (
            point_labels == -1
        )

        for i in range(self.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.prompt_encoder.point_embeddings[i].weight * (
                point_labels == i
            )

        return point_embedding

    def _get_dense_pe(self):
        """Get positional encoding for image embeddings."""
        return self.prompt_encoder.get_dense_pe()

    def select_masks(self, masks, iou_preds, num_points):
        """Select best mask based on IoU predictions."""
        score_reweight = torch.tensor(
            [[1000] + [0] * (self.mask_decoder.num_mask_tokens - 1)]
        ).to(iou_preds.device)
        score = iou_preds + (num_points - 2.5) * score_reweight
        best_idx = torch.argmax(score, dim=1)
        masks = masks[torch.arange(masks.shape[0]), best_idx, :, :].unsqueeze(1)
        iou_preds = iou_preds[torch.arange(masks.shape[0]), best_idx].unsqueeze(1)
        return masks, iou_preds

    @torch.no_grad()
    def forward(self, image_embeddings, point_coords, point_labels):
        """
        Decode masks from embeddings and prompts.

        Args:
            image_embeddings: (B, 256, 64, 64) from encoder
            point_coords: (B, N, 2) point coordinates in image space
            point_labels: (B, N) point labels (1=foreground, 0=background)

        Returns:
            masks: Low-resolution masks (B, 1, 256, 256)
            iou_predictions: IoU scores (B, 1)
        """
        # Embed points
        sparse_embedding = self._embed_points(point_coords, point_labels)

        # No mask input - use the no_mask_embed
        dense_embedding = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            image_embeddings.shape[0], -1, *self.prompt_encoder.image_embedding_size
        )

        # Predict masks
        masks, scores = self.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self._get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
        )

        if self.return_single_mask:
            masks, scores = self.select_masks(masks, scores, point_coords.shape[1])

        return masks, scores


def export_encoder(model, output_path, image_size=1024, device="cpu"):
    """Export the SAM1 image encoder to ONNX format."""
    print(f"Exporting encoder to {output_path}...")

    # Create encoder wrapper
    encoder = SAM1EncoderONNX(model)
    encoder.to(device)
    encoder.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32, device=device)

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
            dynamic_axes={"image": {0: "batch_size"}, "image_embeddings": {0: "batch_size"}},
        )

    print("Encoder exported successfully")


def export_decoder(model, output_path, device="cpu"):
    """Export the SAM1 mask decoder to ONNX format."""
    print(f"Exporting decoder to {output_path}...")

    # Get embedding dimensions from the model
    embed_dim = model.prompt_encoder.embed_dim
    embed_size = model.prompt_encoder.image_embedding_size

    # Create decoder wrapper
    decoder = SAM1DecoderONNX(model, return_single_mask=True)
    decoder.to(device)
    decoder.eval()

    # Create dummy inputs
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float32, device=device),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float32, device=device),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float32, device=device),
    }

    # Export to ONNX
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        torch.onnx.export(
            decoder,
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
                "iou_predictions": {0: "batch_size"},
            },
        )

    print("Decoder exported successfully")


def main():
    parser = argparse.ArgumentParser(description="Export SAM1 to ONNX for Triton")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to SAM1 checkpoint file (e.g., sam_vit_h_4b8939.pth)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM1 model type (default: vit_h)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model_repository",
        help="Output directory for ONNX models",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Input image size (default: 1024)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for export (cpu or cuda)",
    )

    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    encoder_dir = output_dir / "sam1_encoder" / "1"
    decoder_dir = output_dir / "sam1_decoder" / "1"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    decoder_dir.mkdir(parents=True, exist_ok=True)

    # Build SAM1 model
    print(f"Loading SAM1 model ({args.model_type}) from {args.checkpoint}...")
    model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    model.to(device=args.device)
    model.eval()
    print("Model loaded successfully")

    # Export encoder
    encoder_path = encoder_dir / "model.onnx"
    export_encoder(model, encoder_path, args.image_size, args.device)

    # Export decoder
    decoder_path = decoder_dir / "model.onnx"
    export_decoder(model, decoder_path, args.device)

    print("\nExport complete!")
    print(f"  Encoder: {encoder_path}")
    print(f"  Decoder: {decoder_path}")
    print("\nTo use with Triton, ensure config.pbtxt files exist in:")
    print(f"  {output_dir / 'sam1_encoder' / 'config.pbtxt'}")
    print(f"  {output_dir / 'sam1_decoder' / 'config.pbtxt'}")


if __name__ == "__main__":
    main()
