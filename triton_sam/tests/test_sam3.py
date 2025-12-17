#!/usr/bin/env python3
"""
SAM3 Tracker Testing Script

Tests SAM3 Tracker inference with proper directory organization:
- Input images: test/images/
- Output results: test/output/sam3/
"""

import numpy as np
import cv2
from pathlib import Path

from triton_sam import SAM2TritonClient

# Directory setup
TEST_DIR = Path("test")
IMAGE_DIR = TEST_DIR / "images"
OUTPUT_DIR = TEST_DIR / "output" / "sam3"

# Ensure directories exist
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sigmoid(x):
    """Convert logits to probabilities"""
    return 1 / (1 + np.exp(-x))


def create_overlay(image, mask_logits, color=(0, 255, 0), alpha=0.5):
    """Create colored overlay from mask logits."""
    # Convert logits to binary (logits > 0 means prob > 0.5)
    binary = (mask_logits > 0).astype(np.uint8)

    # Create colored mask
    colored = np.zeros_like(image)
    colored[binary == 1] = color

    # Blend
    overlay = cv2.addWeighted(image, 1-alpha, colored, alpha, 0)

    # Add contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    return overlay, binary


def test_sam3_shapes(image_path):
    """Test SAM3 Tracker segmentation on shapes."""
    print("\n" + "="*70)
    print("Testing SAM3 Tracker Segmentation on Geometric Shapes")
    print("="*70)

    # Load image
    original = cv2.imread(str(image_path))
    print(f"\nImage: {image_path.name}")
    print(f"Size: {original.shape[1]}x{original.shape[0]}")

    # Connect to Triton with SAM3 model
    print("\nConnecting to Triton server with SAM3 Tracker...")
    try:
        client = SAM2TritonClient(model_type="sam3")
        print("✓ Connected to SAM3 Tracker")
    except RuntimeError as e:
        print(f"✗ Failed to connect: {e}")
        print("\nMake sure:")
        print("  1. Triton server is running: docker compose up -d")
        print("  2. SAM3 models are exported: pixi run export-sam3-onnx")
        return

    print("Encoding image with SAM3...")
    client.set_image(str(image_path))

    # Define test points (same as SAM2 test for comparison)
    shapes = [
        {"name": "red_circle", "point": (150, 150), "color": (0, 0, 255)},
        {"name": "blue_rectangle", "point": (375, 175), "color": (255, 0, 0)},
        {"name": "green_triangle", "point": (200, 350), "color": (0, 255, 0)},
    ]

    results = []
    for i, shape in enumerate(shapes, 1):
        print(f"\n[{i}/{len(shapes)}] Segmenting {shape['name']} with SAM3...")
        print(f"  Click point: {shape['point']}")

        # Run inference
        masks, iou = client.predict(
            point_coords=np.array([shape['point']]),
            point_labels=np.array([1])
        )

        # SAM3 returns 3 masks per prediction: [batch, num_prompts, 3, H, W]
        # and 3 IoU scores: [batch, num_prompts, 3]
        # Select the best mask based on highest IoU
        print(f"  Mask shape: {masks.shape}")
        print(f"  IoU shape: {iou.shape}")

        iou_scores = iou[0, 0]  # Shape: (3,)
        best_mask_idx = np.argmax(iou_scores)  # Index of best mask
        print(f"  IoU scores (all 3 masks): {iou_scores}")
        print(f"  Best mask index: {best_mask_idx}")

        # Use the mask with highest IoU
        mask_logits = masks[0, 0, best_mask_idx]
        mask_resized = cv2.resize(mask_logits, (original.shape[1], original.shape[0]))

        iou_score = float(iou_scores[best_mask_idx])
        print(f"  Using IoU: {iou_score:.3f}")

        # Create visualizations
        overlay, binary = create_overlay(original, mask_resized, shape['color'])

        # Save outputs
        prefix = OUTPUT_DIR / shape['name']
        cv2.imwrite(f"{prefix}_binary.png", binary * 255)
        cv2.imwrite(f"{prefix}_overlay.png", overlay)

        # Probability map
        prob_map = sigmoid(mask_resized)
        cv2.imwrite(f"{prefix}_probability.png", (prob_map * 255).astype(np.uint8))

        results.append({
            "name": shape['name'],
            "mask": binary,
            "color": shape['color'],
            "iou": iou_score
        })

    # Create combined visualization
    combined = np.zeros_like(original)
    for result in results:
        combined[result["mask"] == 1] = result["color"]

    final_overlay = cv2.addWeighted(original, 0.6, combined, 0.4, 0)

    # Save combined results
    cv2.imwrite(str(OUTPUT_DIR / "all_masks_combined.png"), combined)
    cv2.imwrite(str(OUTPUT_DIR / "all_masks_overlay.png"), final_overlay)

    # Print summary
    print("\n" + "="*70)
    print("Results Summary (SAM3 Tracker)")
    print("="*70)
    print("\nSegmentation Quality:")
    for result in results:
        print(f"  {result['name']:20s} IoU: {result['iou']:.3f}")

    print(f"\nOutput Files:")
    print(f"  📁 {OUTPUT_DIR}/")
    print("     📄 all_masks_combined.png    - All masks colored")
    print("     📄 all_masks_overlay.png     - Overlay on original")
    for result in results:
        print(f"     📄 {result['name']}_*.png")

    print("\n✅ SAM3 Tracker test complete!")
    print("\nTo compare with SAM2, run: pixi run test-sam2")


def main():
    # Use existing test image from SAM2 tests
    test_img_path = IMAGE_DIR / "test_shapes.jpg"

    if not test_img_path.exists():
        print("Error: Test image not found!")
        print(f"Expected: {test_img_path}")
        print("\nPlease run SAM2 test first to generate test image:")
        print("  pixi run test-sam2")
        return

    # Run tests
    test_sam3_shapes(test_img_path)


if __name__ == "__main__":
    main()
