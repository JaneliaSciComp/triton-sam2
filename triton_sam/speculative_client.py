"""
Speculative SAM2 Triton Client with Request Cancellation

Supports queuing many inference requests and cancelling them when no longer needed.
Perfect for interactive segmentation where the user moves the mouse rapidly.

This client provides asynchronous request handling with:
- Request ID tracking by session
- Bulk cancellation of pending requests
- Thread-safe request management
"""

import uuid
import asyncio
import threading
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
import numpy as np
import cv2
import tritonclient.http as httpclient
import tritonclient.http.aio as async_httpclient


class SpeculativeSAM2Client:
    """
    SAM2 client with request tracking and cancellation support.

    Features:
    - Asynchronous request submission
    - Request ID tracking by session
    - Bulk cancellation of pending requests
    - Thread-safe request management
    """

    def __init__(self, url="localhost:8000", timeout=30):
        """Initialize the client."""
        self.url = url
        self.timeout = timeout
        self.sync_client = httpclient.InferenceServerClient(url=url)
        self.async_client = None  # Created when needed

        # Request tracking
        self.pending_requests: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.request_lock = threading.Lock()

        # Image state (cached embeddings)
        self.current_embeddings = None
        self.original_size = None
        self.resized_size = (1024, 1024)

    async def _get_async_client(self):
        """Get or create async client."""
        if self.async_client is None:
            self.async_client = async_httpclient.InferenceServerClient(url=self.url)
        return self.async_client

    def set_image(self, image_path: str) -> np.ndarray:
        """
        Encode image and cache embeddings (synchronous).

        Args:
            image_path: Path to input image

        Returns:
            Image embeddings
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        self.original_size = image.shape[:2]

        # Resize to 1024x1024
        image_resized = cv2.resize(image, self.resized_size)

        # Convert BGR to RGB and normalize
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_tensor = image_rgb.astype(np.float32) / 255.0

        # Add batch dimension and transpose to (1, 3, H, W)
        image_tensor = np.transpose(image_tensor, (2, 0, 1))
        image_tensor = np.expand_dims(image_tensor, axis=0)

        # Run encoder
        inputs = [httpclient.InferInput("image", image_tensor.shape, "FP32")]
        inputs[0].set_data_from_numpy(image_tensor)

        outputs = [httpclient.InferRequestedOutput("image_embeddings")]

        response = self.sync_client.infer(
            model_name="sam2_encoder",
            inputs=inputs,
            outputs=outputs
        )

        self.current_embeddings = response.as_numpy("image_embeddings")
        return self.current_embeddings

    def _prepare_decoder_inputs(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray
    ) -> List[httpclient.InferInput]:
        """Prepare decoder inputs with coordinate scaling."""
        if self.current_embeddings is None:
            raise RuntimeError("Must call set_image() before predict()")

        # Scale coordinates from original image space to model space (1024x1024)
        point_coords = point_coords.copy().astype(np.float32)
        if len(point_coords.shape) == 2:
            point_coords = np.expand_dims(point_coords, axis=0)
        if len(point_labels.shape) == 1:
            point_labels = np.expand_dims(point_labels, axis=0)

        scale_x = 1024.0 / self.original_size[1]
        scale_y = 1024.0 / self.original_size[0]
        point_coords[:, :, 0] *= scale_x
        point_coords[:, :, 1] *= scale_y

        point_labels = point_labels.astype(np.float32)

        # Create inputs
        inputs = [
            httpclient.InferInput("image_embeddings", self.current_embeddings.shape, "FP32"),
            httpclient.InferInput("point_coords", point_coords.shape, "FP32"),
            httpclient.InferInput("point_labels", point_labels.shape, "FP32"),
        ]

        inputs[0].set_data_from_numpy(self.current_embeddings)
        inputs[1].set_data_from_numpy(point_coords)
        inputs[2].set_data_from_numpy(point_labels)

        return inputs

    async def predict_async(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Tuple[str, asyncio.Task]:
        """
        Submit asynchronous prediction request.

        Args:
            point_coords: Point coordinates (N, 2)
            point_labels: Point labels (N,) - 1=foreground, 0=background
            session_id: Optional session identifier for grouping requests
            request_id: Optional specific request ID (auto-generated if None)

        Returns:
            (request_id, task) tuple - task is the async inference task
        """
        if request_id is None:
            request_id = str(uuid.uuid4())
        if session_id is None:
            session_id = "default"

        # Track request
        with self.request_lock:
            self.pending_requests[session_id][request_id] = "pending"

        # Prepare inputs
        inputs = self._prepare_decoder_inputs(point_coords, point_labels)
        outputs = [
            async_httpclient.InferRequestedOutput("masks"),
            async_httpclient.InferRequestedOutput("iou_predictions")
        ]

        # Create async task
        client = await self._get_async_client()

        async def run_inference():
            try:
                response = await client.infer(
                    model_name="sam2_decoder",
                    inputs=inputs,
                    outputs=outputs,
                    request_id=request_id
                )

                # Mark as completed
                with self.request_lock:
                    if request_id in self.pending_requests.get(session_id, {}):
                        self.pending_requests[session_id][request_id] = "completed"

                masks = response.as_numpy("masks")
                iou = response.as_numpy("iou_predictions")
                return masks, iou

            except Exception as e:
                # Mark as failed
                with self.request_lock:
                    if request_id in self.pending_requests.get(session_id, {}):
                        self.pending_requests[session_id][request_id] = "failed"
                raise e

        task = asyncio.create_task(run_inference())
        return request_id, task

    def predict_sync(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synchronous prediction (for comparison/testing).

        Args:
            point_coords: Point coordinates (N, 2)
            point_labels: Point labels (N,)

        Returns:
            (masks, iou_predictions) tuple
        """
        inputs = self._prepare_decoder_inputs(point_coords, point_labels)
        outputs = [
            httpclient.InferRequestedOutput("masks"),
            httpclient.InferRequestedOutput("iou_predictions")
        ]

        response = self.sync_client.infer(
            model_name="sam2_decoder",
            inputs=inputs,
            outputs=outputs
        )

        masks = response.as_numpy("masks")
        iou = response.as_numpy("iou_predictions")
        return masks, iou

    def cancel_session_requests(self, session_id: str) -> int:
        """
        Cancel all pending requests for a session.

        Note: Triton's HTTP API doesn't support true cancellation of in-flight
        requests. This method marks requests as cancelled locally so results
        can be ignored when they complete.

        Args:
            session_id: Session ID to cancel

        Returns:
            Number of requests marked as cancelled
        """
        cancelled_count = 0
        with self.request_lock:
            if session_id in self.pending_requests:
                for request_id, status in self.pending_requests[session_id].items():
                    if status == "pending":
                        self.pending_requests[session_id][request_id] = "cancelled"
                        cancelled_count += 1
        return cancelled_count

    def get_session_status(self, session_id: str) -> Dict[str, int]:
        """
        Get statistics for a session.

        Args:
            session_id: Session ID to query

        Returns:
            Dictionary with counts by status
        """
        status_counts = defaultdict(int)
        with self.request_lock:
            if session_id in self.pending_requests:
                for status in self.pending_requests[session_id].values():
                    status_counts[status] += 1
        return dict(status_counts)

    def cleanup_session(self, session_id: str):
        """Remove all tracking data for a session."""
        with self.request_lock:
            if session_id in self.pending_requests:
                del self.pending_requests[session_id]


# Async helper functions for easier usage

async def queue_multiple_requests(
    client: SpeculativeSAM2Client,
    point_coords_list: List[np.ndarray],
    point_labels_list: List[np.ndarray],
    session_id: str
) -> List[Tuple[str, asyncio.Task]]:
    """
    Queue multiple prediction requests asynchronously.

    Args:
        client: SpeculativeSAM2Client instance
        point_coords_list: List of point coordinate arrays
        point_labels_list: List of point label arrays
        session_id: Session ID for grouping

    Returns:
        List of (request_id, task) tuples
    """
    tasks = []
    for coords, labels in zip(point_coords_list, point_labels_list):
        request_id, task = await client.predict_async(
            coords, labels, session_id=session_id
        )
        tasks.append((request_id, task))
    return tasks


async def wait_for_latest_result(
    tasks: List[Tuple[str, asyncio.Task]],
    client: SpeculativeSAM2Client,
    session_id: str
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Wait for the latest request to complete, ignore cancelled ones.

    Args:
        tasks: List of (request_id, task) tuples
        client: SpeculativeSAM2Client instance
        session_id: Session ID

    Returns:
        (masks, iou) from the latest non-cancelled request, or None
    """
    if not tasks:
        return None

    # Wait for the last task (latest request)
    latest_id, latest_task = tasks[-1]

    try:
        result = await latest_task

        # Check if it was cancelled
        with client.request_lock:
            status = client.pending_requests.get(session_id, {}).get(latest_id)
            if status == "cancelled":
                return None

        return result
    except Exception as e:
        print(f"Request {latest_id} failed: {e}")
        return None
