#!/usr/bin/env python3
"""
SAM2 Pipeline Timing Benchmark

This script measures the latency of each step in the SAM2 image segmentation pipeline.
It runs N iterations and reports mean latency for each step on CPU and/or GPU.

SAM2 Architecture Components:
=============================

The Segment Anything Model 2 (SAM2) consists of the following components:

1. IMAGE ENCODER (Hiera Backbone)
   - Hierarchical vision transformer that processes RGB images
   - Extracts multi-scale spatial feature embeddings
   - Most expensive operation but cached for interactive workflows

2. PROMPT ENCODER (Geometric)
   - Encodes spatial prompts (points, boxes) into prompt tokens
   - Uses positional embeddings and learned encodings
   - Supports foreground/background point labels

3. MASK DECODER
   - Cross-attention between image features and prompt tokens
   - Generates high-resolution segmentation mask logits
   - Produces pixel-wise predictions from coarse features

Note: Unlike SAM3, SAM2 does NOT support text prompts. It only accepts
geometric prompts (points and boxes) for segmentation.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

# Add sam2_repo to path
sam2_repo_path = Path(__file__).parent.parent / "sam2_repo"
sys.path.insert(0, str(sam2_repo_path))

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm


class TimingStats:
    """Helper class to collect and compute timing statistics."""

    def __init__(self, name: str):
        self.name = name
        self.times: List[float] = []

    def add(self, duration: float):
        """Add a timing measurement in milliseconds."""
        self.times.append(duration)

    def mean(self) -> float:
        """Get mean latency in milliseconds."""
        return np.mean(self.times) if self.times else 0.0

    def std(self) -> float:
        """Get standard deviation in milliseconds."""
        return np.std(self.times) if self.times else 0.0

    def min(self) -> float:
        """Get minimum latency in milliseconds."""
        return np.min(self.times) if self.times else 0.0

    def max(self) -> float:
        """Get maximum latency in milliseconds."""
        return np.max(self.times) if self.times else 0.0


def visualize_results(
    image,
    masks,
    box_prompt: List[float] = None,
    point_coords: List[List[float]] = None,
    point_labels: List[int] = None,
    output_path: str = "sam2_result.png"
):
    """
    Visualize segmentation results and save to file.

    Args:
        image: PIL Image
        masks: Segmentation masks (numpy array)
        box_prompt: Box prompt in XYXY format
        point_coords: Point coordinates
        point_labels: Point labels (1=foreground, 0=background)
        output_path: Path to save visualization
    """
    all_masks = None
    num_masks = 0

    if masks is not None:
        # Convert masks to numpy if needed
        if isinstance(masks, torch.Tensor):
            masks_np = masks.cpu().numpy()
        else:
            masks_np = np.array(masks)

        # Process masks if not empty
        if masks_np.size > 0:
            # Handle different mask shapes
            # SAM2 returns masks with shape (C, H, W) where C is number of masks
            if masks_np.ndim == 3 and masks_np.shape[0] > 0:
                all_masks = masks_np
                num_masks = all_masks.shape[0]
            elif masks_np.ndim == 2:
                all_masks = masks_np[np.newaxis, ...]
                num_masks = 1

    print(f"Found {num_masks} mask(s) to visualize")

    # Define distinct colors for different instances (RGBA)
    instance_colors = [
        [0.12, 0.56, 1.0, 0.6],   # Blue
        [1.0, 0.27, 0.0, 0.6],    # Orange-red
        [0.0, 0.8, 0.4, 0.6],     # Green
        [0.8, 0.0, 0.8, 0.6],     # Magenta
        [1.0, 0.84, 0.0, 0.6],    # Gold
        [0.0, 0.8, 0.8, 0.6],     # Cyan
        [0.6, 0.3, 0.0, 0.6],     # Brown
        [0.5, 0.0, 0.5, 0.6],     # Purple
        [0.0, 0.5, 0.0, 0.6],     # Dark green
        [1.0, 0.41, 0.71, 0.6],   # Hot pink
    ]

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Original image with prompts
    axes[0].imshow(image)
    axes[0].set_title("Input Image with Prompts", fontsize=14)
    axes[0].axis("off")

    # Draw prompts
    if point_coords and point_labels:
        for (x, y), label in zip(point_coords, point_labels):
            color = 'green' if label == 1 else 'red'
            marker = '*'
            axes[0].scatter(x, y, color=color, marker=marker, s=500,
                          edgecolor='white', linewidth=2, zorder=10)

    if box_prompt:
        x1, y1, x2, y2 = box_prompt
        w, h = x2 - x1, y2 - y1
        rect = mpatches.Rectangle((x1, y1), w, h, linewidth=2,
                                 edgecolor='green', facecolor='none')
        axes[0].add_patch(rect)

    # Add prompt info as text
    prompt_text = []
    if point_coords:
        n_pos = sum(1 for l in point_labels if l == 1) if point_labels else 0
        n_neg = sum(1 for l in point_labels if l == 0) if point_labels else 0
        prompt_text.append(f"Points: {n_pos} positive, {n_neg} negative")
    if box_prompt:
        prompt_text.append("Box prompt")

    if prompt_text:
        axes[0].text(0.02, 0.98, '\n'.join(prompt_text),
                    transform=axes[0].transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Right: Image with mask overlay
    axes[1].imshow(image)

    # Overlay all masks with different colors (if any)
    if all_masks is not None:
        for i, mask in enumerate(all_masks):
            # Threshold at 0 for logits or 0.5 for binary masks
            threshold = 0.0 if mask.min() < 0 else 0.5
            color_mask = np.zeros((*mask.shape, 4))
            color = instance_colors[i % len(instance_colors)]
            color_mask[mask > threshold] = color
            axes[1].imshow(color_mask)

    axes[1].set_title(f"Segmentation Result ({num_masks} mask{'s' if num_masks != 1 else ''})", fontsize=14)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {output_path}")


def time_step(func, sync_cuda: bool = False):
    """Time a function call, optionally syncing CUDA."""
    start = time.perf_counter()
    result = func()
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    duration_ms = (end - start) * 1000  # Convert to milliseconds
    return result, duration_ms


# =============================================================================
# VIDEO MODE FUNCTIONS
# =============================================================================

# Define distinct colors for video overlay (BGR format for cv2)
VIDEO_COLORS = [
    (255, 144, 30),    # Blue
    (0, 69, 255),      # Orange-red
    (102, 204, 0),     # Green
    (204, 0, 204),     # Magenta
    (0, 215, 255),     # Gold
    (204, 204, 0),     # Cyan
    (0, 76, 153),      # Brown
    (128, 0, 128),     # Purple
    (0, 128, 0),       # Dark green
    (180, 105, 255),   # Hot pink
]


def create_mask_overlay_cv2(frame: np.ndarray, masks: torch.Tensor, alpha: float = 0.5) -> np.ndarray:
    """
    Create a colored mask overlay on a video frame.

    Args:
        frame: BGR frame from cv2 (H, W, 3)
        masks: Mask tensor (num_objects, H, W) with values > 0 for foreground
        alpha: Overlay transparency (0-1)

    Returns:
        Frame with mask overlay (H, W, 3)
    """
    overlay = frame.copy()

    if masks is None:
        return overlay

    # Convert to numpy if needed
    if isinstance(masks, torch.Tensor):
        masks_np = masks.cpu().numpy()
    else:
        masks_np = masks

    # Handle different shapes
    if masks_np.ndim == 2:
        masks_np = masks_np[np.newaxis, ...]
    elif masks_np.ndim == 4:
        # (batch, num_objects, H, W) -> (num_objects, H, W)
        masks_np = masks_np.squeeze(0)

    # Apply each mask with a different color
    for i, mask in enumerate(masks_np):
        color = VIDEO_COLORS[i % len(VIDEO_COLORS)]
        binary_mask = (mask > 0).astype(np.uint8)

        # Resize mask to frame size if needed
        if binary_mask.shape != frame.shape[:2]:
            binary_mask = cv2.resize(binary_mask, (frame.shape[1], frame.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)

        # Create colored overlay
        colored = np.zeros_like(frame)
        colored[binary_mask == 1] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, alpha, 0)

        # Add contours for better visibility
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    return overlay


def benchmark_sam2_video(
    video_path: str,
    output_video_path: str = None,
    device: str = "cuda",
    box_prompt: List[float] = None,
    point_coords: List[List[float]] = None,
    point_labels: List[int] = None,
    model_checkpoint: str = None,
    model_cfg: str = None,
    alpha: float = 0.5,
):
    """
    Run SAM2 video segmentation benchmark.

    Applies prompts on the first frame and propagates through the video.

    Args:
        video_path: Path to input MP4 video
        output_video_path: Path to output video (default: sam2_output.mp4)
        device: Device to run on ("cuda" or "cpu")
        box_prompt: Box prompt in XYXY format [x1, y1, x2, y2]
        point_coords: Point coordinates [[x1, y1], [x2, y2], ...]
        point_labels: Point labels (1=foreground, 0=background)
        model_checkpoint: Path to model checkpoint
        model_cfg: Path to model config
        alpha: Mask overlay transparency (0-1)
    """
    print(f"\n{'='*80}")
    print(f"SAM2 Video Pipeline Benchmark")
    print(f"{'='*80}")
    print(f"Device: {device.upper()}")
    print(f"Input video: {video_path}")

    # Determine prompt type
    if box_prompt and point_coords:
        prompt_type = "Box + Points"
    elif box_prompt:
        prompt_type = "Box"
    elif point_coords:
        prompt_type = "Points"
    else:
        print("ERROR: No prompts provided. Use -p/--points or -b/--box-prompt")
        return

    print(f"Prompt type: {prompt_type}")
    if box_prompt:
        print(f"Box prompt (XYXY): {box_prompt}")
    if point_coords:
        print(f"Point coords: {point_coords}")
        print(f"Point labels: {point_labels}")
    print(f"{'='*80}\n")

    # Set device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    # Setup for CUDA
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    # Determine paths
    script_dir = Path(__file__).parent.parent

    if model_checkpoint is None:
        model_checkpoint = str(script_dir / "checkpoints" / "sam2.1_hiera_base_plus.pt")

    if model_cfg is None:
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

    if output_video_path is None:
        output_video_path = "sam2_output.mp4"

    # Timing stats
    timings = {}
    sync_cuda = (device == "cuda" and torch.cuda.is_available())

    # =========================================================================
    # STEP 1: Load Video Predictor
    # =========================================================================
    print("Loading SAM2 video predictor...")
    model_start = time.perf_counter()
    predictor = build_sam2_video_predictor(model_cfg, model_checkpoint, device=device)
    if sync_cuda:
        torch.cuda.synchronize()
    timings["load_model"] = (time.perf_counter() - model_start) * 1000
    print(f"Model loaded in {timings['load_model']:.2f} ms\n")

    # =========================================================================
    # STEP 2: Initialize Video State
    # =========================================================================
    print("Initializing video state...")
    init_start = time.perf_counter()
    inference_state = predictor.init_state(video_path=video_path)
    if sync_cuda:
        torch.cuda.synchronize()
    timings["init_video"] = (time.perf_counter() - init_start) * 1000

    num_frames = inference_state["num_frames"]
    video_height = inference_state["video_height"]
    video_width = inference_state["video_width"]
    print(f"Video initialized: {num_frames} frames, {video_width}x{video_height}")
    print(f"Init time: {timings['init_video']:.2f} ms\n")

    # =========================================================================
    # STEP 3: Add Prompts on First Frame
    # =========================================================================
    print("Adding prompts on frame 0...")
    prompt_start = time.perf_counter()

    # Prepare prompts
    points = np.array(point_coords, dtype=np.float32) if point_coords else None
    labels = np.array(point_labels, dtype=np.int32) if point_labels else None
    box = np.array(box_prompt, dtype=np.float32) if box_prompt else None

    # Add prompts to frame 0 with object ID 1
    frame_idx, obj_ids, mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=points,
        labels=labels,
        box=box,
    )
    if sync_cuda:
        torch.cuda.synchronize()
    timings["add_prompts"] = (time.perf_counter() - prompt_start) * 1000
    print(f"Prompts added in {timings['add_prompts']:.2f} ms\n")

    # =========================================================================
    # STEP 4: Propagate Through Video
    # =========================================================================
    print("Propagating masks through video...")
    prop_start = time.perf_counter()

    # Collect all frame masks
    video_segments = {}
    for frame_idx, obj_ids, mask_logits in tqdm(
        predictor.propagate_in_video(inference_state),
        total=num_frames,
        desc="Propagating"
    ):
        video_segments[frame_idx] = {
            "obj_ids": obj_ids,
            "masks": (mask_logits > 0.0).cpu()  # Convert logits to binary masks
        }

    if sync_cuda:
        torch.cuda.synchronize()
    timings["propagation"] = (time.perf_counter() - prop_start) * 1000
    timings["propagation_per_frame"] = timings["propagation"] / num_frames
    print(f"\nPropagation complete: {timings['propagation']:.2f} ms total")
    print(f"Per-frame average: {timings['propagation_per_frame']:.2f} ms\n")

    # =========================================================================
    # STEP 5: Write Output Video with Overlays
    # =========================================================================
    print(f"Writing output video to: {output_video_path}")
    write_start = time.perf_counter()

    # Open input video to read frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0  # Default FPS

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (video_width, video_height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get masks for this frame
        if frame_idx in video_segments:
            masks = video_segments[frame_idx]["masks"]
            frame_with_overlay = create_mask_overlay_cv2(frame, masks, alpha=alpha)
        else:
            frame_with_overlay = frame

        out.write(frame_with_overlay)
        frame_idx += 1

    cap.release()
    out.release()

    timings["write_video"] = (time.perf_counter() - write_start) * 1000
    print(f"Video written in {timings['write_video']:.2f} ms\n")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    timings["total_pipeline"] = sum(v for k, v in timings.items() if k != "propagation_per_frame")

    print("="*80)
    print(f"RESULTS - SAM2 VIDEO BENCHMARK ({device.upper()})")
    print("="*80)
    print(f"{'Step':<30} {'Time (ms)':<15}")
    print("-"*80)
    print(f"{'Load Model':<30} {timings['load_model']:<15.2f}")
    print(f"{'Initialize Video':<30} {timings['init_video']:<15.2f}")
    print(f"{'Add Prompts (Frame 0)':<30} {timings['add_prompts']:<15.2f}")
    print(f"{'Propagation (Total)':<30} {timings['propagation']:<15.2f}")
    print(f"{'Propagation (Per Frame)':<30} {timings['propagation_per_frame']:<15.2f}")
    print(f"{'Write Output Video':<30} {timings['write_video']:<15.2f}")
    print("-"*80)
    print(f"{'TOTAL PIPELINE':<30} {timings['total_pipeline']:<15.2f}")
    print("="*80)
    print(f"\nOutput: {output_video_path}")
    print(f"Frames: {num_frames}, FPS: {fps:.1f}")
    print()

    return timings


def run_sam2_pipeline_with_timing(
    predictor: SAM2ImagePredictor,
    image_path: str,
    box_prompt: List[float] = None,
    point_coords: List[List[float]] = None,
    point_labels: List[int] = None,
    device: str = "cuda",
    multimask_output: bool = False,
    return_outputs: bool = False
) -> Dict[str, float]:
    """
    Run the full SAM2 pipeline with timing for each step.

    SAM2 Architecture Overview:
    ===========================
    1. Image Encoder (Hiera backbone) - Encodes RGB image into spatial features
    2. Prompt Encoder - Encodes geometric prompts (points, boxes) into tokens
    3. Mask Decoder - Generates segmentation masks from image + prompt features

    Note: SAM2 does NOT support text prompts (unlike SAM3).

    Args:
        predictor: SAM2ImagePredictor instance
        image_path: Path to input image
        box_prompt: Box prompt in XYXY format [x1, y1, x2, y2]
        point_coords: List of point coordinates [[x1, y1], [x2, y2], ...]
        point_labels: List of point labels (1=foreground, 0=background)
        device: Device to run on ("cuda" or "cpu")
        multimask_output: If True, return 3 masks with quality scores
        return_outputs: If True, return (timings, image, masks)

    Returns a dictionary mapping step names to duration in milliseconds.
    """
    timings = {}
    sync_cuda = (device == "cuda" and torch.cuda.is_available())

    # ============================================================================
    # STEP 1: Load Image (I/O - not a SAM2 component)
    # ============================================================================
    # Simply loads the RGB image from disk into PIL/numpy format.
    # This is preprocessing, not part of the SAM2 neural architecture.

    def load_image():
        img = Image.open(image_path)
        return np.array(img.convert("RGB"))

    image_np, t = time_step(load_image, sync_cuda)
    timings["load_image"] = t

    # ============================================================================
    # STEP 2: Set Image - IMAGE ENCODER (Hiera Backbone)
    # ============================================================================
    # SAM2 Component: IMAGE ENCODER
    #
    # This is the most expensive operation in the pipeline. The Image Encoder
    # (Hiera hierarchical vision transformer backbone) processes the input RGB
    # image and extracts dense spatial feature embeddings at multiple scales.
    #
    # Input:  RGB image (H, W, 3)
    # Output: Multi-scale feature maps (stored in predictor._features)
    #
    # These image features are cached in the predictor and reused for
    # all subsequent mask predictions with different prompts, enabling fast
    # interactive segmentation workflows.

    def set_image():
        predictor.set_image(image_np)

    _, t = time_step(set_image, sync_cuda)
    timings["set_image_encode"] = t

    # ============================================================================
    # STEP 3: Predict Masks - PROMPT ENCODER + MASK DECODER
    # ============================================================================
    # This step involves the remaining SAM2 components:
    #
    # PROMPT ENCODER: Encodes spatial prompts into prompt tokens:
    #   - Point prompts: xy coordinates + label (foreground/background)
    #   - Box prompts: corner coordinates defining ROI (XYXY format)
    #   These are encoded with positional embeddings and learned encodings
    #
    # MASK DECODER: Cross-attends between:
    #   - Image features from the Image Encoder
    #   - Prompt tokens from the Prompt Encoder
    #   Generates high-resolution segmentation mask logits
    #
    # The predict() call executes both components in sequence.

    masks = None
    iou_predictions = None

    if box_prompt and point_coords:
        # ========================================================================
        # Box + Points Workflow (Combined prompting)
        # ========================================================================
        # Uses both box and point prompts for refined segmentation.
        # The box provides the region of interest, points refine the mask.

        def predict_with_box_and_points():
            return predictor.predict(
                point_coords=np.array(point_coords),
                point_labels=np.array(point_labels),
                box=np.array(box_prompt),
                multimask_output=multimask_output,
                return_logits=True,
            )

        (masks, iou_predictions, _), t = time_step(predict_with_box_and_points, sync_cuda)
        timings["predict_box_and_points"] = t

    elif box_prompt:
        # ========================================================================
        # Box-Only Workflow
        # ========================================================================
        # Uses PROMPT ENCODER (box) + MASK DECODER
        # Box defines the region of interest in XYXY format

        def predict_with_box():
            return predictor.predict(
                box=np.array(box_prompt),
                multimask_output=multimask_output,
                return_logits=True,
            )

        (masks, iou_predictions, _), t = time_step(predict_with_box, sync_cuda)
        timings["predict_box"] = t

    elif point_coords:
        # ========================================================================
        # Point-Only Workflow
        # ========================================================================
        # Uses PROMPT ENCODER (points) + MASK DECODER
        # Points indicate foreground (1) or background (0) regions

        def predict_with_points():
            return predictor.predict(
                point_coords=np.array(point_coords),
                point_labels=np.array(point_labels),
                multimask_output=multimask_output,
                return_logits=True,
            )

        (masks, iou_predictions, _), t = time_step(predict_with_points, sync_cuda)
        timings["predict_points"] = t

    # ============================================================================
    # PIPELINE SUMMARY
    # ============================================================================
    # Total pipeline time = Image Encoding + Prompt Processing + Mask Generation
    #
    # For interactive workflows:
    # - Image Encoding (Step 2): Run ONCE per image (~300ms GPU, ~2000ms CPU)
    # - Mask Prediction (Step 3): Run MANY times with different prompts (~15ms GPU)
    #
    # The cached image embeddings enable real-time interactive segmentation
    # where users can try different points or boxes without re-encoding the image.

    timings["total_pipeline"] = sum(timings.values())

    if return_outputs:
        # Convert numpy image back to PIL for visualization
        pil_image = Image.fromarray(image_np)
        return timings, pil_image, masks, iou_predictions
    return timings


def benchmark_sam2(
    n_iterations: int = 1,
    device: str = "cuda",
    box_prompt: List[float] = None,
    point_coords: List[List[float]] = None,
    point_labels: List[int] = None,
    image_path: str = None,
    model_checkpoint: str = None,
    model_cfg: str = None,
    multimask_output: bool = False,
    visualize: bool = False,
    output_dir: str = "."
):
    """
    Run SAM2 benchmark for N iterations and report statistics.

    Args:
        n_iterations: Number of iterations to run
        device: Device to run on ("cuda" or "cpu")
        box_prompt: Box prompt in XYXY format [x1, y1, x2, y2]
        point_coords: Point coordinates [[x1, y1], [x2, y2], ...]
        point_labels: Point labels (1=foreground, 0=background)
        image_path: Path to input image
        model_checkpoint: Path to model checkpoint
        model_cfg: Path to model config file
        multimask_output: If True, return 3 masks with quality scores
        visualize: If True, save visualization of results
        output_dir: Directory to save visualizations
    """
    print(f"\n{'='*80}")
    print(f"SAM2 Pipeline Benchmark")
    print(f"{'='*80}")
    print(f"Device: {device.upper()}")
    print(f"Iterations: {n_iterations}")

    # Determine prompt type
    if box_prompt and point_coords:
        prompt_type = "Box + Points"
    elif box_prompt:
        prompt_type = "Box"
    elif point_coords:
        prompt_type = "Points"
    else:
        prompt_type = "None"

    print(f"Prompt type: {prompt_type}")
    if box_prompt:
        print(f"Box prompt (XYXY): {box_prompt}")
    if point_coords:
        print(f"Point coords: {point_coords}")
        print(f"Point labels: {point_labels}")
    if visualize:
        print(f"Visualization: Enabled (output to {output_dir}/)")
    print(f"{'='*80}\n")

    # Set device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    # Setup for CUDA
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    # Determine paths
    script_dir = Path(__file__).parent.parent

    if model_checkpoint is None:
        model_checkpoint = str(script_dir / "checkpoints" / "sam2.1_hiera_base_plus.pt")

    if model_cfg is None:
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

    # Load model (only once, outside timing)
    print("Loading SAM2 model...")
    model_start = time.perf_counter()
    model = build_sam2(model_cfg, model_checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)
    model_load_time = (time.perf_counter() - model_start) * 1000

    print(f"Model loaded in {model_load_time:.2f} ms\n")

    # Sync CUDA if needed
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    # Default image path
    if image_path is None:
        image_path = str(script_dir / "sam2_repo" / "notebooks" / "images" / "truck.jpg")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"Using image: {image_path}")
    print(f"\nRunning {n_iterations} iterations...\n")

    # Initialize timing collectors
    stats = {
        "load_image": TimingStats("Load Image"),
        "set_image_encode": TimingStats("Set Image (Encode)"),
        "predict_box": TimingStats("Predict (Box)"),
        "predict_points": TimingStats("Predict (Points)"),
        "predict_box_and_points": TimingStats("Predict (Box + Points)"),
        "total_pipeline": TimingStats("Total Pipeline"),
    }

    # Run benchmark iterations
    saved_outputs = None
    for i in range(n_iterations):
        print(f"Iteration {i+1}/{n_iterations}...", end="\r")

        # Reset predictor state between iterations
        predictor.reset_predictor()

        # On last iteration, optionally return outputs for visualization
        return_outputs = visualize and (i == n_iterations - 1)

        result = run_sam2_pipeline_with_timing(
            predictor=predictor,
            image_path=image_path,
            box_prompt=box_prompt,
            point_coords=point_coords,
            point_labels=point_labels,
            device=device,
            multimask_output=multimask_output,
            return_outputs=return_outputs
        )

        if return_outputs:
            timings, image, masks, iou_predictions = result
            saved_outputs = (image, masks)
        else:
            timings = result

        # Collect statistics
        for step, duration in timings.items():
            stats[step].add(duration)

    print(" " * 50, end="\r")  # Clear the progress line

    # Print results
    print("\n" + "="*80)
    print(f"RESULTS - {device.upper()} ({n_iterations} iterations)")
    print("="*80)
    print(f"{'Step':<30} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
    print("-"*80)

    # Order of steps to display
    step_order = [
        "load_image",
        "set_image_encode",
    ]

    # Add appropriate prediction step
    if box_prompt and point_coords:
        step_order.append("predict_box_and_points")
    elif box_prompt:
        step_order.append("predict_box")
    elif point_coords:
        step_order.append("predict_points")

    step_order.append("total_pipeline")

    for step in step_order:
        if step in stats and stats[step].times:
            s = stats[step]
            print(f"{s.name:<30} {s.mean():<12.2f} {s.std():<12.2f} {s.min():<12.2f} {s.max():<12.2f}")

    print("="*80)
    print()

    # Generate visualization if requested
    if visualize and saved_outputs is not None:
        print(f"\nGenerating visualization...")
        image, masks = saved_outputs

        # Create output filename
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "sam2_result.png")

        visualize_results(
            image=image,
            masks=masks,
            box_prompt=box_prompt,
            point_coords=point_coords,
            point_labels=point_labels,
            output_path=output_path
        )
        print()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SAM2 pipeline with timing for each step"
    )
    parser.add_argument(
        "-n", "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run (default: 1)"
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to run on (default: cuda)"
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run benchmarks on both CPU and GPU"
    )
    parser.add_argument(
        "-b", "--box-prompt",
        type=float,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Box prompt in XYXY format (e.g., 100 100 400 300)"
    )
    parser.add_argument(
        "-p", "--points",
        type=float,
        nargs="+",
        metavar="COORD",
        help="Point coordinates as x1 y1 x2 y2 ... (e.g., 512 512 600 400)"
    )
    parser.add_argument(
        "-l", "--point-labels",
        type=int,
        nargs="+",
        metavar="LABEL",
        help="Point labels (1=foreground, 0=background) (e.g., 1 1)"
    )
    parser.add_argument(
        "-i", "--image",
        type=str,
        help="Path to input image (default: sam2_repo/notebooks/images/truck.jpg)"
    )
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        help="Path to model checkpoint (default: checkpoints/sam2.1_hiera_base_plus.pt)"
    )
    parser.add_argument(
        "--model-cfg",
        type=str,
        help="Path to model config (default: configs/sam2.1/sam2.1_hiera_b+.yaml)"
    )
    parser.add_argument(
        "--multimask",
        action="store_true",
        help="Return 3 masks with quality scores instead of single best mask"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization of results"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=".",
        help="Directory to save visualizations (default: current directory)"
    )

    # Video mode arguments
    parser.add_argument(
        "-v", "--video",
        type=str,
        help="Path to MP4 video file (enables video mode)"
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Output video path (default: sam2_output.mp4)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Mask overlay transparency for video mode (0-1, default: 0.5)"
    )

    args = parser.parse_args()

    # Parse point coordinates
    point_coords = None
    if args.points:
        if len(args.points) % 2 != 0:
            parser.error("Point coordinates must be pairs of x,y values")
        point_coords = [[args.points[i], args.points[i+1]]
                       for i in range(0, len(args.points), 2)]

    # Parse point labels
    point_labels = args.point_labels if args.point_labels else None

    # Validate points and labels match
    if point_coords and point_labels:
        if len(point_coords) != len(point_labels):
            parser.error("Number of points and labels must match")
    elif point_coords and not point_labels:
        # Default to all foreground points
        point_labels = [1] * len(point_coords)

    # Check for video mode
    if args.video:
        # VIDEO MODE
        if not os.path.exists(args.video):
            parser.error(f"Video file not found: {args.video}")

        benchmark_sam2_video(
            video_path=args.video,
            output_video_path=args.output_video,
            device=args.device,
            box_prompt=args.box_prompt,
            point_coords=point_coords,
            point_labels=point_labels,
            model_checkpoint=args.checkpoint,
            model_cfg=args.model_cfg,
            alpha=args.alpha,
        )
        return

    # IMAGE MODE - Run benchmark
    if args.both:
        # Run on both CPU and GPU
        benchmark_sam2(
            n_iterations=args.iterations,
            device="cuda",
            box_prompt=args.box_prompt,
            point_coords=point_coords,
            point_labels=point_labels,
            image_path=args.image,
            model_checkpoint=args.checkpoint,
            model_cfg=args.model_cfg,
            multimask_output=args.multimask,
            visualize=args.visualize,
            output_dir=args.output_dir
        )

        benchmark_sam2(
            n_iterations=args.iterations,
            device="cpu",
            box_prompt=args.box_prompt,
            point_coords=point_coords,
            point_labels=point_labels,
            image_path=args.image,
            model_checkpoint=args.checkpoint,
            model_cfg=args.model_cfg,
            multimask_output=args.multimask,
            visualize=args.visualize,
            output_dir=args.output_dir
        )
    else:
        benchmark_sam2(
            n_iterations=args.iterations,
            device=args.device,
            box_prompt=args.box_prompt,
            point_coords=point_coords,
            point_labels=point_labels,
            image_path=args.image,
            model_checkpoint=args.checkpoint,
            model_cfg=args.model_cfg,
            multimask_output=args.multimask,
            visualize=args.visualize,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
