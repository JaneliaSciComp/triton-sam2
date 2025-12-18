#!/usr/bin/env python3
"""
SAM3 Pipeline Timing Benchmark

This script measures the latency of each step in the SAM3 image segmentation pipeline.
It runs N iterations and reports mean latency for each step on CPU and/or GPU.

SAM3 Architecture Components (from paper):
==========================================

The Segment Anything Model 3 (SAM3) consists of the following components:

1. IMAGE ENCODER (Hiera Backbone)
   - Hierarchical vision transformer that processes RGB images
   - Extracts multi-scale spatial feature embeddings
   - Most expensive operation but cached for interactive workflows

2. TEXT ENCODER (CLIP-style)
   - Encodes natural language prompts into semantic embeddings
   - Uses tokenization and transformer architecture
   - Enables open-vocabulary segmentation

3. DETECTOR (Grounding Module)
   - Performs open-vocabulary object detection
   - Cross-attends between text embeddings and image features
   - Localizes objects matching text descriptions
   - Outputs bounding boxes and confidence scores

4. PROMPT ENCODER (Geometric)
   - Encodes spatial prompts (points, boxes) into prompt tokens
   - Uses positional embeddings and learned encodings
   - Supports foreground/background point labels

5. MASK DECODER
   - Cross-attention between image features and prompt tokens
   - Generates high-resolution segmentation mask logits
   - Produces pixel-wise predictions from coarse features

6. MASK MERGING & REFINEMENT
   - Combines multiple masks from different prompts/detections
   - Applies post-processing (NMS, boundary refinement)
   - Filters masks by confidence threshold

7. TRACKER (Video mode - not used in single-image benchmarks)
   - Tracks objects across video frames
   - Maintains object identity over time

8. MEMORY BANK (Video mode - not used in single-image benchmarks)
   - Stores embeddings for temporal consistency
   - Enables efficient video segmentation

This benchmark focuses on single-image segmentation, measuring components 1-6.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

# Add sam3_repo to path
sam3_repo_path = Path(__file__).parent.parent / "sam3_repo"
sys.path.insert(0, str(sam3_repo_path))

import sam3
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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
    text_prompt: str = None,
    box_prompt: List[float] = None,
    point_coords: List[List[float]] = None,
    point_labels: List[int] = None,
    output_path: str = "sam3_result.png"
):
    """
    Visualize segmentation results and save to file.

    Args:
        image: PIL Image
        masks: Segmentation masks tensor
        text_prompt: Text prompt used
        box_prompt: Box prompt in xywh format
        point_coords: Point coordinates
        point_labels: Point labels (1=foreground, 0=background)
        output_path: Path to save visualization
    """
    all_masks = None
    num_masks = 0

    if masks is not None:
        # Convert masks to numpy
        if isinstance(masks, torch.Tensor):
            masks_np = masks.cpu().numpy()
        else:
            masks_np = np.array(masks)

        # Process masks if not empty
        if masks_np.size > 0:
            # Handle different mask shapes
            # SAM3 returns masks with shape (N, 1, H, W) where N is number of detections
            # and 1 is a channel dimension that should be squeezed out
            if masks_np.ndim == 4:  # (num_detections, 1, H, W)
                if masks_np.shape[0] > 0:
                    # Squeeze out the channel dimension (axis 1), keep all detections
                    masks_np = masks_np.squeeze(axis=1)  # -> (N, H, W)

            # Handle different mask dimensionalities
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
        x, y, w, h = box_prompt
        rect = mpatches.Rectangle((x, y), w, h, linewidth=2,
                                 edgecolor='green', facecolor='none')
        axes[0].add_patch(rect)

    # Add prompt info as text
    prompt_text = []
    if text_prompt:
        prompt_text.append(f"Text: '{text_prompt}'")
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

    axes[1].set_title(f"Segmentation Result ({num_masks} instance{'s' if num_masks != 1 else ''})", fontsize=14)
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


def run_sam3_pipeline_with_timing(
    model,
    image_path: str,
    text_prompt: str = None,
    box_prompt: List[float] = None,
    point_coords: List[List[float]] = None,
    point_labels: List[int] = None,
    device: str = "cuda",
    confidence_threshold: float = 0.3,
    return_outputs: bool = False
) -> Dict[str, float]:
    """
    Run the full SAM3 pipeline with timing for each step.

    SAM3 Architecture Overview (from paper):
    ========================================
    1. Image Encoder (Hiera backbone) - Encodes RGB image into spatial features
    2. Text Encoder (CLIP-style) - Encodes text prompts into semantic embeddings
    3. Detector (Grounding) - Detects objects matching text/geometric prompts
    4. Prompt Encoder - Encodes geometric prompts (points, boxes) into tokens
    5. Mask Decoder - Generates segmentation masks from image + prompt features
    6. Mask Merging - Combines multiple masks and refines boundaries
    7. Tracker - Tracks objects across video frames (not used for single images)
    8. Memory Bank - Stores embeddings for temporal consistency (not used for single images)

    Args:
        model: SAM3 model
        image_path: Path to input image
        text_prompt: Text prompt (e.g., "shoe")
        box_prompt: Box prompt in xywh format
        point_coords: List of point coordinates [[x1, y1], [x2, y2], ...]
        point_labels: List of point labels (1=foreground, 0=background)
        device: Device to run on ("cuda" or "cpu")
        return_outputs: If True, return (timings, image, masks, inference_state)

    Returns a dictionary mapping step names to duration in milliseconds.
    """
    timings = {}
    sync_cuda = (device == "cuda" and torch.cuda.is_available())

    # ============================================================================
    # STEP 1: Load Image (I/O - not a SAM3 component)
    # ============================================================================
    # Simply loads the RGB image from disk into PIL format.
    # This is preprocessing, not part of the SAM3 neural architecture.

    def load_image():
        return Image.open(image_path)

    image, t = time_step(load_image, sync_cuda)
    timings["load_image"] = t
    width, height = image.size

    # ============================================================================
    # STEP 2: Create Processor (Pipeline setup - not a SAM3 component)
    # ============================================================================
    # Initializes the Sam3Processor which wraps the model and manages the
    # inference pipeline. Sets confidence threshold for detection filtering.

    def create_processor():
        return Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)

    processor, t = time_step(create_processor, sync_cuda)
    timings["create_processor"] = t

    # ============================================================================
    # STEP 3: Set Image - IMAGE ENCODER (Hiera Backbone)
    # ============================================================================
    # SAM3 Component: IMAGE ENCODER
    #
    # This is the most expensive operation in the pipeline. The Image Encoder
    # (Hiera hierarchical vision transformer backbone) processes the input RGB
    # image and extracts dense spatial feature embeddings at multiple scales.
    #
    # Input:  RGB image (H, W, 3)
    # Output: Multi-scale feature maps (e.g., 256 channels at 64x64 resolution)
    #
    # These image features are cached in the inference_state and reused for
    # all subsequent mask predictions with different prompts, enabling fast
    # interactive segmentation workflows.

    def set_image():
        return processor.set_image(image)

    inference_state, t = time_step(set_image, sync_cuda)
    timings["set_image_encode"] = t

    # ============================================================================
    # STEP 4: Add Prompts - TEXT ENCODER + DETECTOR + PROMPT ENCODER + MASK DECODER
    # ============================================================================
    # This step involves multiple SAM3 components working together:
    #
    # For Text Prompts:
    #   1. TEXT ENCODER: Encodes text prompt (e.g., "shoe") into semantic embeddings
    #      using a CLIP-style text encoder with vocabulary tokenization
    #
    #   2. DETECTOR (Grounding): Performs open-vocabulary object detection by
    #      cross-attending between text embeddings and image features to localize
    #      objects matching the text description. Outputs bounding boxes and scores.
    #
    # For Geometric Prompts (Points/Boxes):
    #   3. PROMPT ENCODER: Encodes spatial prompts into prompt tokens:
    #      - Point prompts: xy coordinates + label (foreground/background)
    #      - Box prompts: corner coordinates defining ROI
    #      These are encoded with positional embeddings and learned encodings
    #
    # After Prompt Encoding:
    #   4. MASK DECODER: Cross-attends between:
    #      - Image features from the Image Encoder
    #      - Prompt tokens from Text/Geometric Encoder
    #      - Detected object features (if using text grounding)
    #      Generates high-resolution segmentation mask logits
    #
    #   5. MASK MERGING: Combines multiple detected objects/prompts and refines
    #      mask boundaries using post-processing and NMS (non-maximum suppression)
    #
    # The _forward_grounding() call executes all these components in sequence.

    if text_prompt and point_coords:
        # ========================================================================
        # Text + Points Workflow (Combined prompting)
        # ========================================================================
        # Uses both TEXT ENCODER + DETECTOR for grounding AND PROMPT ENCODER
        # for geometric refinement. The point prompts refine the text-grounded
        # detections by indicating which regions to include/exclude.

        def add_text_and_points():
            processor.reset_all_prompts(inference_state)
            # TEXT ENCODER + DETECTOR: Encode text and detect objects
            state = processor.set_text_prompt(state=inference_state, prompt=text_prompt)

            # PROMPT ENCODER: Add geometric point prompts for refinement
            if point_coords and point_labels:
                # Normalize points to [0, 1] range
                norm_points = torch.tensor(point_coords, dtype=torch.float32)
                norm_points[:, 0] = norm_points[:, 0] / width
                norm_points[:, 1] = norm_points[:, 1] / height

                # Add points to geometric prompt encoder
                labels_tensor = torch.tensor(point_labels, dtype=torch.long, device=device)
                state["geometric_prompt"].input_points = norm_points.unsqueeze(0).to(device)
                state["geometric_prompt"].input_points_mask = torch.ones(
                    (1, len(point_coords)), dtype=torch.bool, device=device
                )
                # Store labels for visualization (positive=1, negative=0)
                # SAM3 uses different encoding internally

            # MASK DECODER + MASK MERGING: Generate final segmentation masks
            return processor._forward_grounding(state)

        inference_state, t = time_step(add_text_and_points, sync_cuda)
        timings["add_text_and_points"] = t

    elif text_prompt:
        # ========================================================================
        # Text-Only Workflow
        # ========================================================================
        # Uses TEXT ENCODER + DETECTOR + MASK DECODER + MASK MERGING
        # Performs open-vocabulary segmentation without geometric constraints

        def add_text_prompt():
            processor.reset_all_prompts(inference_state)
            # TEXT ENCODER + DETECTOR + MASK DECODER + MASK MERGING
            return processor.set_text_prompt(state=inference_state, prompt=text_prompt)

        inference_state, t = time_step(add_text_prompt, sync_cuda)
        timings["add_text_prompt"] = t

    elif box_prompt:
        # ========================================================================
        # Box-Only Workflow
        # ========================================================================
        # Uses PROMPT ENCODER (box) + MASK DECODER
        # No text encoding or detection, purely geometric prompting

        box_input_xywh = torch.tensor(box_prompt).view(-1, 4)
        box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)
        norm_box_cxcywh = normalize_bbox(box_input_cxcywh, width, height).flatten().tolist()

        def add_box_prompt():
            processor.reset_all_prompts(inference_state)
            # PROMPT ENCODER (box) + MASK DECODER
            return processor.add_geometric_prompt(
                state=inference_state, box=norm_box_cxcywh, label=True
            )

        inference_state, t = time_step(add_box_prompt, sync_cuda)
        timings["add_box_prompt"] = t

    elif point_coords:
        # ========================================================================
        # Point-Only Workflow
        # ========================================================================
        # Uses PROMPT ENCODER (points) + MASK DECODER
        # No text encoding or detection, purely geometric prompting
        # Uses "visual" as dummy text to make model rely on geometric prompts

        def add_point_prompt():
            processor.reset_all_prompts(inference_state)

            # Set dummy text prompt to enable geometric-only mode
            dummy_text_outputs = processor.model.backbone.forward_text(
                ["visual"], device=device
            )
            inference_state["backbone_out"].update(dummy_text_outputs)

            if "geometric_prompt" not in inference_state:
                inference_state["geometric_prompt"] = processor.model._get_dummy_prompt()

            # Normalize points to [0, 1] range
            norm_points = torch.tensor(point_coords, dtype=torch.float32)
            norm_points[:, 0] = norm_points[:, 0] / width
            norm_points[:, 1] = norm_points[:, 1] / height

            # Add points to geometric prompt encoder
            inference_state["geometric_prompt"].input_points = norm_points.unsqueeze(0).to(device)
            inference_state["geometric_prompt"].input_points_mask = torch.ones(
                (1, len(point_coords)), dtype=torch.bool, device=device
            )

            # MASK DECODER + MASK MERGING: Generate final segmentation masks
            return processor._forward_grounding(inference_state)

        inference_state, t = time_step(add_point_prompt, sync_cuda)
        timings["add_point_prompt"] = t

    # ============================================================================
    # STEP 5: Get Masks (Retrieval - not a SAM3 component)
    # ============================================================================
    # This step simply retrieves the already-computed segmentation masks from
    # the inference_state dictionary. The actual mask generation happened in
    # Step 4 during the MASK DECODER and MASK MERGING stages.
    #
    # The masks are stored as logits (pre-sigmoid values) which need to be
    # thresholded at 0.0 to produce binary segmentation masks:
    #   - mask_logits > 0.0  => foreground (object)
    #   - mask_logits <= 0.0 => background
    #
    # This step is essentially zero-cost (just dictionary access).

    def get_masks():
        # The masks are already in inference_state after prompting
        # Use masks_logits which contains all masks before thresholding
        masks = inference_state.get("masks_logits", None)
        if masks is None:
            masks = inference_state.get("masks", None)
        return masks

    masks, t = time_step(get_masks, sync_cuda)
    timings["get_masks"] = t

    # ============================================================================
    # PIPELINE SUMMARY
    # ============================================================================
    # Total pipeline time = Image Encoding + Prompt Processing + Mask Generation
    #
    # For interactive workflows:
    # - Image Encoding (Step 3): Run ONCE per image (~2000ms CPU, ~40ms GPU)
    # - Prompt Processing (Step 4): Run MANY times with different prompts
    #                                (~300ms CPU, ~40ms GPU per prompt)
    #
    # The cached image embeddings enable real-time interactive segmentation
    # where users can try different text prompts or click different points
    # without re-encoding the image.

    timings["total_pipeline"] = sum(timings.values())

    if return_outputs:
        return timings, image, masks, inference_state
    return timings


def benchmark_sam3(
    n_iterations: int = 1,
    device: str = "cuda",
    text_prompt: str = None,
    box_prompt: List[float] = None,
    point_coords: List[List[float]] = None,
    point_labels: List[int] = None,
    image_path: str = None,
    model_checkpoint: str = None,
    confidence_threshold: float = 0.3,
    visualize: bool = False,
    output_dir: str = "."
):
    """
    Run SAM3 benchmark for N iterations and report statistics.

    Args:
        n_iterations: Number of iterations to run
        device: Device to run on ("cuda" or "cpu")
        text_prompt: Text prompt (e.g., "shoe")
        box_prompt: Box prompt in xywh format
        point_coords: Point coordinates [[x1, y1], [x2, y2], ...]
        point_labels: Point labels (1=foreground, 0=background)
        image_path: Path to input image
        model_checkpoint: Path to model checkpoint
        visualize: If True, save visualization of results
        output_dir: Directory to save visualizations
    """
    print(f"\n{'='*80}")
    print(f"SAM3 Pipeline Benchmark")
    print(f"{'='*80}")
    print(f"Device: {device.upper()}")
    print(f"Iterations: {n_iterations}")

    # Determine prompt type
    if text_prompt and point_coords:
        prompt_type = "Text + Points"
    elif text_prompt:
        prompt_type = "Text"
    elif box_prompt:
        prompt_type = "Box"
    elif point_coords:
        prompt_type = "Points"
    else:
        prompt_type = "None"

    print(f"Prompt type: {prompt_type}")
    print(f"Confidence threshold: {confidence_threshold}")
    if text_prompt:
        print(f"Text prompt: '{text_prompt}'")
    if box_prompt:
        print(f"Box prompt: {box_prompt}")
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

    # Load model (only once, outside timing)
    print("Loading SAM3 model...")
    sam3_root = Path(sam3.__file__).parent.parent

    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    model_start = time.perf_counter()
    if model_checkpoint:
        model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=model_checkpoint, device=device)
    else:
        model = build_sam3_image_model(bpe_path=bpe_path, device=device)
    model_load_time = (time.perf_counter() - model_start) * 1000

    print(f"Model loaded in {model_load_time:.2f} ms\n")

    # Sync CUDA if needed (model already loaded to correct device)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    # Default image path
    if image_path is None:
        image_path = f"{sam3_root}/assets/images/test_image.jpg"

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"Using image: {image_path}")
    print(f"\nRunning {n_iterations} iterations...\n")

    # Initialize timing collectors
    stats = {
        "load_image": TimingStats("Load Image"),
        "create_processor": TimingStats("Create Processor"),
        "set_image_encode": TimingStats("Set Image (Encode)"),
        "add_text_prompt": TimingStats("Add Text Prompt"),
        "add_text_and_points": TimingStats("Add Text + Points"),
        "add_box_prompt": TimingStats("Add Box Prompt"),
        "add_point_prompt": TimingStats("Add Point Prompt"),
        "get_masks": TimingStats("Get Masks"),
        "total_pipeline": TimingStats("Total Pipeline"),
    }

    # Run benchmark iterations
    saved_outputs = None
    for i in range(n_iterations):
        print(f"Iteration {i+1}/{n_iterations}...", end="\r")

        # On last iteration, optionally return outputs for visualization
        return_outputs = visualize and (i == n_iterations - 1)

        result = run_sam3_pipeline_with_timing(
            model=model,
            image_path=image_path,
            text_prompt=text_prompt,
            box_prompt=box_prompt,
            point_coords=point_coords,
            point_labels=point_labels,
            device=device,
            confidence_threshold=confidence_threshold,
            return_outputs=return_outputs
        )

        if return_outputs:
            timings, image, masks, inference_state = result
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
    print(f"{'Step':<25} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
    print("-"*80)

    # Order of steps to display
    step_order = [
        "load_image",
        "create_processor",
        "set_image_encode",
    ]

    # Add appropriate prompt step
    if text_prompt and point_coords:
        step_order.append("add_text_and_points")
    elif text_prompt:
        step_order.append("add_text_prompt")
    elif box_prompt:
        step_order.append("add_box_prompt")
    elif point_coords:
        step_order.append("add_point_prompt")

    step_order.extend(["get_masks", "total_pipeline"])

    for step in step_order:
        if step in stats and stats[step].times:
            s = stats[step]
            print(f"{s.name:<25} {s.mean():<12.2f} {s.std():<12.2f} {s.min():<12.2f} {s.max():<12.2f}")

    print("="*80)
    print()

    # Generate visualization if requested
    if visualize and saved_outputs is not None:
        print(f"\nGenerating visualization...")
        image, masks = saved_outputs

        # Create output filename
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "sam3_result.png")

        visualize_results(
            image=image,
            masks=masks,
            text_prompt=text_prompt,
            box_prompt=box_prompt,
            point_coords=point_coords,
            point_labels=point_labels,
            output_path=output_path
        )
        print()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SAM3 pipeline with timing for each step"
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
        "-t", "--text-prompt",
        type=str,
        default=None,
        help="Text prompt for segmentation"
    )
    parser.add_argument(
        "-b", "--box-prompt",
        type=float,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        help="Box prompt in xywh format (e.g., 480 290 110 360)"
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
        help="Path to input image (default: SAM3 test image)"
    )
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Confidence threshold for detections (default: 0.3)"
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

    # Determine text prompt (None if box-only prompt)
    text_prompt = None if (args.box_prompt and not point_coords) else args.text_prompt

    # Run benchmark
    if args.both:
        # Run on both CPU and GPU
        benchmark_sam3(
            n_iterations=args.iterations,
            device="cuda",
            text_prompt=text_prompt,
            box_prompt=args.box_prompt,
            point_coords=point_coords,
            point_labels=point_labels,
            image_path=args.image,
            model_checkpoint=args.checkpoint,
            confidence_threshold=args.confidence,
            visualize=args.visualize,
            output_dir=args.output_dir
        )

        benchmark_sam3(
            n_iterations=args.iterations,
            device="cpu",
            text_prompt=text_prompt,
            box_prompt=args.box_prompt,
            point_coords=point_coords,
            point_labels=point_labels,
            image_path=args.image,
            model_checkpoint=args.checkpoint,
            confidence_threshold=args.confidence,
            visualize=args.visualize,
            output_dir=args.output_dir
        )
    else:
        benchmark_sam3(
            n_iterations=args.iterations,
            device=args.device,
            text_prompt=text_prompt,
            box_prompt=args.box_prompt,
            point_coords=point_coords,
            point_labels=point_labels,
            image_path=args.image,
            model_checkpoint=args.checkpoint,
            confidence_threshold=args.confidence,
            visualize=args.visualize,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
