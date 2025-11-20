#!/usr/bin/env python3
"""
SAM2 Triton Inference Client

This client demonstrates how to use the SAM2 models deployed on Triton
to perform interactive image segmentation.
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


class SAM2TritonClient:
    """Client for SAM2 inference using Triton Inference Server."""

    def __init__(self, triton_url="localhost:8000"):
        """Initialize the client.

        Args:
            triton_url: URL of the Triton server (default: localhost:8000)
        """
        self.client = httpclient.InferenceServerClient(url=triton_url)
        self.image_embeddings = None
        self.original_size = None

        # Verify server is ready
        if not self.client.is_server_ready():
            raise RuntimeError("Triton server is not ready")

        # Verify models are loaded
        if not self.client.is_model_ready("sam2_encoder"):
            raise RuntimeError("sam2_encoder model is not ready")
        if not self.client.is_model_ready("sam2_decoder"):
            raise RuntimeError("sam2_decoder model is not ready")

        print("✓ Connected to Triton server")
        print("✓ Models loaded successfully")

    def set_image(self, image_path, image_size=1024):
        """Encode an image and cache its embeddings.

        Args:
            image_path: Path to the input image
            image_size: Size to resize image to (default: 1024)
        """
        # Load and preprocess image
        print(f"Loading image: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        self.original_size = image.shape[:2]
        print(f"Original size: {self.original_size}")

        # Resize to model input size
        image = cv2.resize(image, (image_size, image_size))
        # Convert BGR to RGB and normalize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        # Transpose to CHW format
        image = np.transpose(image, (2, 0, 1))
        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        print(f"Preprocessed shape: {image.shape}")

        # Create input
        encoder_input = httpclient.InferInput("image", image.shape, "FP32")
        encoder_input.set_data_from_numpy(image)

        # Request output
        encoder_output = httpclient.InferRequestedOutput("image_embeddings")

        # Run inference
        print("Encoding image...")
        try:
            response = self.client.infer(
                "sam2_encoder",
                [encoder_input],
                outputs=[encoder_output]
            )
            self.image_embeddings = response.as_numpy("image_embeddings")
            print(f"✓ Image encoded, embedding shape: {self.image_embeddings.shape}")

        except InferenceServerException as e:
            raise RuntimeError(f"Encoder inference failed: {e}")

    def predict(self, point_coords, point_labels):
        """Predict segmentation mask from point prompts.

        Args:
            point_coords: Array of [x, y] coordinates in ORIGINAL image space, shape (N, 2)
            point_labels: Array of labels (1=foreground, 0=background), shape (N,)

        Returns:
            masks: Segmentation masks, shape (1, H, W)
            iou_predictions: IoU confidence scores
        """
        if self.image_embeddings is None:
            raise RuntimeError("Must call set_image() before predict()")

        # Prepare inputs
        point_coords = np.array(point_coords, dtype=np.float32)
        point_labels = np.array(point_labels, dtype=np.float32)

        # IMPORTANT: Scale coordinates from original image space to model input space (1024x1024)
        scale_x = 1024.0 / self.original_size[1]  # width
        scale_y = 1024.0 / self.original_size[0]  # height
        point_coords[:, 0] *= scale_x
        point_coords[:, 1] *= scale_y

        # Add batch dimension
        point_coords = np.expand_dims(point_coords, axis=0)
        point_labels = np.expand_dims(point_labels, axis=0)

        print(f"Predicting with {len(point_coords[0])} points...")

        # Create decoder inputs
        decoder_inputs = [
            httpclient.InferInput(
                "image_embeddings",
                self.image_embeddings.shape,
                "FP32"
            ),
            httpclient.InferInput("point_coords", point_coords.shape, "FP32"),
            httpclient.InferInput("point_labels", point_labels.shape, "FP32"),
        ]

        decoder_inputs[0].set_data_from_numpy(self.image_embeddings)
        decoder_inputs[1].set_data_from_numpy(point_coords)
        decoder_inputs[2].set_data_from_numpy(point_labels)

        # Request outputs
        decoder_outputs = [
            httpclient.InferRequestedOutput("masks"),
            httpclient.InferRequestedOutput("iou_predictions"),
        ]

        # Run inference
        try:
            response = self.client.infer(
                "sam2_decoder",
                decoder_inputs,
                outputs=decoder_outputs
            )

            masks = response.as_numpy("masks")
            iou = response.as_numpy("iou_predictions")

            print(f"✓ Prediction complete, IoU: {float(iou.flat[0]):.3f}")

            return masks, iou

        except InferenceServerException as e:
            raise RuntimeError(f"Decoder inference failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="SAM2 Triton Inference Client"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--points",
        type=str,
        required=True,
        help="Point coordinates as x1,y1,x2,y2,... (e.g., 512,512)"
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Point labels as 1,0,1,... (1=fg, 0=bg). Defaults to all 1s"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_mask.png",
        help="Output mask file path"
    )
    parser.add_argument(
        "--triton-url",
        type=str,
        default="localhost:8000",
        help="Triton server URL"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization with points and mask overlay"
    )

    args = parser.parse_args()

    # Parse points
    coords = [float(x) for x in args.points.split(",")]
    if len(coords) % 2 != 0:
        raise ValueError("Points must be in x,y pairs")

    point_coords = [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]

    # Parse labels
    if args.labels:
        point_labels = [int(x) for x in args.labels.split(",")]
        if len(point_labels) != len(point_coords):
            raise ValueError("Number of labels must match number of points")
    else:
        point_labels = [1] * len(point_coords)  # Default to foreground

    print(f"Points: {point_coords}")
    print(f"Labels: {point_labels}")

    # Create client and run inference
    client = SAM2TritonClient(args.triton_url)
    client.set_image(args.image)
    masks, iou = client.predict(point_coords, point_labels)

    # Save mask
    mask = masks[0, 0] > 0  # Threshold at 0
    mask_image = (mask * 255).astype(np.uint8)

    # Resize to original image size if needed
    if client.original_size:
        mask_image = cv2.resize(
            mask_image,
            (client.original_size[1], client.original_size[0]),
            interpolation=cv2.INTER_LINEAR
        )

    cv2.imwrite(args.output, mask_image)
    print(f"✓ Mask saved to {args.output}")

    # Create visualization if requested
    if args.visualize:
        image = cv2.imread(args.image)
        # Resize mask to match original image
        mask_resized = cv2.resize(
            mask_image,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

        # Create colored overlay
        overlay = image.copy()
        overlay[mask_resized > 128] = [0, 255, 0]  # Green mask
        result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

        # Draw points
        for (x, y), label in zip(point_coords, point_labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)  # Green=fg, Red=bg
            cv2.circle(result, (int(x), int(y)), 10, color, -1)
            cv2.circle(result, (int(x), int(y)), 12, (255, 255, 255), 2)

        vis_path = args.output.replace(".png", "_vis.png")
        cv2.imwrite(vis_path, result)
        print(f"✓ Visualization saved to {vis_path}")


if __name__ == "__main__":
    main()
