#!/usr/bin/env python3
"""
Comprehensive SAM2 Testing Script

Tests SAM2 inference with proper directory organization:
- Input images: test/images/
- Output results: test/output/
"""

import numpy as np
import cv2
from pathlib import Path

from triton_sam2 import SAM2TritonClient

# Directory setup
TEST_DIR = Path("test")
IMAGE_DIR = TEST_DIR / "images"
OUTPUT_DIR = TEST_DIR / "output"

# Ensure directories exist
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sigmoid(x):
    """Convert logits to probabilities"""
    return 1 / (1 + np.exp(-x))


def create_test_image():
    """Create a test image with geometric shapes."""
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255

    # Red circle
    cv2.circle(img, (150, 150), 80, (0, 0, 255), -1)
    # Blue rectangle
    cv2.rectangle(img, (300, 100), (450, 250), (255, 0, 0), -1)
    # Green triangle
    pts = np.array([[100, 400], [200, 250], [300, 400]], np.int32)
    cv2.fillPoly(img, [pts], (0, 255, 0))

    return img


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


def test_shapes(image_path):
    """Test segmentation on shapes."""
    print("\n" + "="*70)
    print("Testing SAM2 Segmentation on Geometric Shapes")
    print("="*70)

    # Load image
    original = cv2.imread(str(image_path))
    print(f"\nImage: {image_path.name}")
    print(f"Size: {original.shape[1]}x{original.shape[0]}")

    # Connect to Triton
    print("\nConnecting to Triton server...")
    client = SAM2TritonClient()

    print("Encoding image...")
    client.set_image(str(image_path))

    # Define test points
    shapes = [
        {"name": "red_circle", "point": (150, 150), "color": (0, 0, 255)},
        {"name": "blue_rectangle", "point": (375, 175), "color": (255, 0, 0)},
        {"name": "green_triangle", "point": (200, 350), "color": (0, 255, 0)},
    ]

    results = []
    for i, shape in enumerate(shapes, 1):
        print(f"\n[{i}/{len(shapes)}] Segmenting {shape['name']}...")
        print(f"  Click point: {shape['point']}")

        # Run inference
        masks, iou = client.predict(
            point_coords=np.array([shape['point']]),
            point_labels=np.array([1])
        )

        # Extract mask and resize
        mask_logits = masks[0, 0]
        mask_resized = cv2.resize(mask_logits, (original.shape[1], original.shape[0]))

        iou_score = float(iou.flat[0])
        print(f"  IoU: {iou_score:.3f}")

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
    print("Results Summary")
    print("="*70)
    print("\nSegmentation Quality:")
    for result in results:
        print(f"  {result['name']:20s} IoU: {result['iou']:.3f}")

    print("\nOutput Files:")
    print(f"  📁 {OUTPUT_DIR}/")
    print("     📄 all_masks_combined.png    - All masks colored")
    print("     📄 all_masks_overlay.png     - Overlay on original")
    for result in results:
        print(f"     📄 {result['name']}_*.png")

    print("\n✅ Test complete!")


def main():
    # Create test image if it doesn't exist
    test_img_path = IMAGE_DIR / "test_shapes.jpg"
    if not test_img_path.exists():
        print("Creating test image with shapes...")
        test_img = create_test_image()
        cv2.imwrite(str(test_img_path), test_img)
        print(f"✓ Saved to {test_img_path}")

    # Run tests
    test_shapes(test_img_path)


if __name__ == "__main__":
    main()
