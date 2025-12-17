"""
SAM Triton Inference Client

Basic synchronous client for SAM2/SAM3 inference using Triton Inference Server.
"""

import numpy as np
import cv2
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


class SAM2TritonClient:
    """
    Client for SAM2/SAM3 inference using Triton Inference Server.

    This client provides a simple synchronous interface for interactive image segmentation.
    It implements the two-stage SAM workflow:
    1. set_image() - Encode image once (expensive, ~300ms)
    2. predict() - Generate masks from prompts (fast, ~15ms)

    Supports both SAM2 and SAM3 Tracker models (SAM3 Tracker is backward compatible with SAM2).

    Example:
        >>> # Use SAM2 (default)
        >>> client = SAM2TritonClient("localhost:8000")
        >>> client.set_image("image.jpg")
        >>> masks, iou = client.predict([[512, 512]], [1])

        >>> # Use SAM3 Tracker
        >>> client = SAM2TritonClient("localhost:8000", model_type="sam3")
        >>> client.set_image("image.jpg")
        >>> masks, iou = client.predict([[512, 512]], [1])
    """

    def __init__(self, triton_url="localhost:8000", model_type="sam2"):
        """
        Initialize the client.

        Args:
            triton_url: URL of the Triton server (default: localhost:8000)
            model_type: Model to use - "sam2" or "sam3" (default: "sam2")
                       SAM3 Tracker is backward compatible with SAM2 API

        Raises:
            RuntimeError: If server or models are not ready
            ValueError: If model_type is invalid
        """
        if model_type not in ["sam2", "sam3"]:
            raise ValueError(f"Invalid model_type: {model_type}. Must be 'sam2' or 'sam3'")

        self.client = httpclient.InferenceServerClient(url=triton_url)
        self.model_type = model_type
        self.encoder_model = f"{model_type}_encoder"
        self.decoder_model = f"{model_type}_decoder"
        self.image_embeddings = None
        self.original_size = None

        # Verify server is ready
        if not self.client.is_server_ready():
            raise RuntimeError("Triton server is not ready")

        # Verify models are loaded
        if not self.client.is_model_ready(self.encoder_model):
            raise RuntimeError(f"{self.encoder_model} model is not ready")
        if not self.client.is_model_ready(self.decoder_model):
            raise RuntimeError(f"{self.decoder_model} model is not ready")

    def set_image(self, image_path, image_size=None):
        """
        Encode an image and cache its embeddings.

        This should be called once per image. The embeddings are cached and reused
        for multiple predict() calls with different prompts.

        Args:
            image_path: Path to the input image
            image_size: Size to resize image to (default: 1024 for SAM2, 1008 for SAM3)

        Raises:
            ValueError: If image cannot be loaded
            RuntimeError: If encoding fails
        """
        # Model-specific image size
        if image_size is None:
            image_size = 1024 if self.model_type == "sam2" else 1008
        self.image_size = image_size

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

        # Model-specific encoder inference
        if self.model_type == "sam2":
            # SAM2: single input "image", single output "image_embeddings"
            encoder_input = httpclient.InferInput("image", image.shape, "FP32")
            encoder_input.set_data_from_numpy(image)
            encoder_output = httpclient.InferRequestedOutput("image_embeddings")

            try:
                response = self.client.infer(
                    self.encoder_model,
                    [encoder_input],
                    outputs=[encoder_output]
                )
                self.image_embeddings = response.as_numpy("image_embeddings")
            except InferenceServerException as e:
                raise RuntimeError(f"Encoder inference failed: {e}")

        else:  # SAM3
            # SAM3: input "pixel_values", three outputs "image_embeddings.0/1/2"
            encoder_input = httpclient.InferInput("pixel_values", image.shape, "FP32")
            encoder_input.set_data_from_numpy(image)
            encoder_outputs = [
                httpclient.InferRequestedOutput("image_embeddings.0"),
                httpclient.InferRequestedOutput("image_embeddings.1"),
                httpclient.InferRequestedOutput("image_embeddings.2"),
            ]

            try:
                response = self.client.infer(
                    self.encoder_model,
                    [encoder_input],
                    outputs=encoder_outputs
                )
                # Store all three embeddings as a tuple
                self.image_embeddings = (
                    response.as_numpy("image_embeddings.0"),
                    response.as_numpy("image_embeddings.1"),
                    response.as_numpy("image_embeddings.2"),
                )
            except InferenceServerException as e:
                raise RuntimeError(f"Encoder inference failed: {e}")

    def predict(self, point_coords, point_labels):
        """
        Predict segmentation mask from point prompts.

        Args:
            point_coords: Array of [x, y] coordinates in ORIGINAL image space, shape (N, 2)
                         Coordinates will be automatically scaled to model space
            point_labels: Array of labels (1=foreground, 0=background), shape (N,)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - masks: Segmentation mask logits
                        SAM2: shape (1, 1, 256, 256)
                        SAM3: shape (1, 1, 3, H, W) - returns 3 masks per prediction
                        Threshold at 0 for binary mask: mask = (logits > 0)
                - iou_predictions: IoU confidence scores
                        SAM2: shape (1, 1)
                        SAM3: shape (1, 1, 3) - 3 IoU scores per mask

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

        # Scale coordinates from original image space to model input space
        scale_x = float(self.image_size) / self.original_size[1]  # width
        scale_y = float(self.image_size) / self.original_size[0]  # height
        point_coords[:, 0] *= scale_x
        point_coords[:, 1] *= scale_y

        # Model-specific decoder inference
        if self.model_type == "sam2":
            # SAM2 format: simple batch dimension
            point_coords = np.expand_dims(point_coords, axis=0)  # (1, N, 2)
            point_labels = np.expand_dims(point_labels, axis=0)  # (1, N)

            decoder_inputs = [
                httpclient.InferInput("image_embeddings", self.image_embeddings.shape, "FP32"),
                httpclient.InferInput("point_coords", point_coords.shape, "FP32"),
                httpclient.InferInput("point_labels", point_labels.shape, "FP32"),
            ]
            decoder_inputs[0].set_data_from_numpy(self.image_embeddings)
            decoder_inputs[1].set_data_from_numpy(point_coords)
            decoder_inputs[2].set_data_from_numpy(point_labels)

            decoder_outputs = [
                httpclient.InferRequestedOutput("masks"),
                httpclient.InferRequestedOutput("iou_predictions"),
            ]

            try:
                response = self.client.infer(
                    self.decoder_model,
                    decoder_inputs,
                    outputs=decoder_outputs
                )
                masks = response.as_numpy("masks")
                iou = response.as_numpy("iou_predictions")
                return masks, iou
            except InferenceServerException as e:
                raise RuntimeError(f"Decoder inference failed: {e}")

        else:  # SAM3
            # SAM3 format: extra dimension for points/labels, INT64 labels, requires boxes
            point_coords = np.expand_dims(point_coords, axis=(0, 1))  # (1, 1, N, 2)
            point_labels = np.expand_dims(point_labels, axis=(0, 1)).astype(np.int64)  # (1, 1, N)

            # SAM3 requires input_boxes even if unused - use zeros
            input_boxes = np.zeros((1, 0, 4), dtype=np.float32)

            # Unpack the three image embeddings
            emb0, emb1, emb2 = self.image_embeddings

            decoder_inputs = [
                httpclient.InferInput("input_points", point_coords.shape, "FP32"),
                httpclient.InferInput("input_labels", point_labels.shape, "INT64"),
                httpclient.InferInput("input_boxes", input_boxes.shape, "FP32"),
                httpclient.InferInput("image_embeddings.0", emb0.shape, "FP32"),
                httpclient.InferInput("image_embeddings.1", emb1.shape, "FP32"),
                httpclient.InferInput("image_embeddings.2", emb2.shape, "FP32"),
            ]
            decoder_inputs[0].set_data_from_numpy(point_coords)
            decoder_inputs[1].set_data_from_numpy(point_labels)
            decoder_inputs[2].set_data_from_numpy(input_boxes)
            decoder_inputs[3].set_data_from_numpy(emb0)
            decoder_inputs[4].set_data_from_numpy(emb1)
            decoder_inputs[5].set_data_from_numpy(emb2)

            decoder_outputs = [
                httpclient.InferRequestedOutput("pred_masks"),
                httpclient.InferRequestedOutput("iou_scores"),
                httpclient.InferRequestedOutput("object_score_logits"),
            ]

            try:
                response = self.client.infer(
                    self.decoder_model,
                    decoder_inputs,
                    outputs=decoder_outputs
                )
                # SAM3 returns pred_masks (5D), iou_scores (3 values), object_score_logits
                masks = response.as_numpy("pred_masks")
                iou = response.as_numpy("iou_scores")
                # Note: ignoring object_score_logits for now, can add if needed
                return masks, iou
            except InferenceServerException as e:
                raise RuntimeError(f"Decoder inference failed: {e}")
