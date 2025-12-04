"""
SAM2 Triton Inference Client

Basic synchronous client for SAM2 inference using Triton Inference Server.
"""

import numpy as np
import cv2
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


class SAM2TritonClient:
    """
    Client for SAM2 inference using Triton Inference Server.

    This client provides a simple synchronous interface for interactive image segmentation.
    It implements the two-stage SAM2 workflow:
    1. set_image() - Encode image once (expensive, ~300ms)
    2. predict() - Generate masks from prompts (fast, ~15ms)

    Example:
        >>> client = SAM2TritonClient("localhost:8000")
        >>> client.set_image("image.jpg")
        >>> masks, iou = client.predict([[512, 512]], [1])  # Click at center
    """

    def __init__(self, triton_url="localhost:8000"):
        """
        Initialize the client.

        Args:
            triton_url: URL of the Triton server (default: localhost:8000)

        Raises:
            RuntimeError: If server or models are not ready
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

    def set_image(self, image_path, image_size=1024):
        """
        Encode an image and cache its embeddings.

        This should be called once per image. The embeddings are cached and reused
        for multiple predict() calls with different prompts.

        Args:
            image_path: Path to the input image
            image_size: Size to resize image to (default: 1024)

        Raises:
            ValueError: If image cannot be loaded
            RuntimeError: If encoding fails
        """
        # Load and preprocess image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        self.original_size = image.shape[:2]

        # Resize to model input size
        image = cv2.resize(image, (image_size, image_size))
        # Convert BGR to RGB and normalize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        # Transpose to CHW format
        image = np.transpose(image, (2, 0, 1))
        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        # Create input
        encoder_input = httpclient.InferInput("image", image.shape, "FP32")
        encoder_input.set_data_from_numpy(image)

        # Request output
        encoder_output = httpclient.InferRequestedOutput("image_embeddings")

        # Run inference
        try:
            response = self.client.infer(
                "sam2_encoder",
                [encoder_input],
                outputs=[encoder_output]
            )
            self.image_embeddings = response.as_numpy("image_embeddings")

        except InferenceServerException as e:
            raise RuntimeError(f"Encoder inference failed: {e}")

    def predict(self, point_coords, point_labels):
        """
        Predict segmentation mask from point prompts.

        Args:
            point_coords: Array of [x, y] coordinates in ORIGINAL image space, shape (N, 2)
                         Coordinates will be automatically scaled to model space (1024x1024)
            point_labels: Array of labels (1=foreground, 0=background), shape (N,)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - masks: Segmentation mask logits, shape (1, 1, 256, 256)
                        Threshold at 0 for binary mask: mask = (logits > 0)
                - iou_predictions: IoU confidence scores, shape (1, 1)

        Raises:
            RuntimeError: If set_image() hasn't been called or inference fails

        Example:
            >>> masks, iou = client.predict([[100, 150], [200, 250]], [1, 0])
            >>> binary_mask = masks[0, 0] > 0  # Threshold logits at 0
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

            return masks, iou

        except InferenceServerException as e:
            raise RuntimeError(f"Decoder inference failed: {e}")
