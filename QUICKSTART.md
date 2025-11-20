# SAM2 + Triton Quick Start

Complete setup from scratch in 3 steps.

## Prerequisites

- NVIDIA GPU with updated drivers (CUDA 12.x)
- [Pixi](https://pixi.sh) installed
- Docker with NVIDIA Container Toolkit

## Installation

### 1. Install Environment & Models

```bash
# Install dependencies
pixi install

# Download model, clone SAM2 repo, export to ONNX
pixi run setup
```

This will:
- Install all Python dependencies
- Download SAM2.1 base_plus checkpoint (~350MB)
- Clone SAM2 repository
- Export encoder and decoder to ONNX format

### 2. Start Triton Server

```bash
docker compose up -d
```

Verify it's running:
```bash
curl http://localhost:8000/v2/health/ready
```

### 3. Run Test

```bash
pixi run test-sam2
```

Results will be in `test/output/`:
- `all_masks_overlay.png` - Visualization of segmented shapes
- Individual masks for each shape

## Project Structure

```
triton_sam/
├── pyproject.toml           # Pixi configuration
├── docker-compose.yml       # Triton server config
├── test_sam2.py            # Test script
├── CLAUDE.md               # Detailed documentation
├── README.md               # Project overview
│
├── model_repository/       # Triton models (auto-generated)
│   ├── sam2_encoder/
│   └── sam2_decoder/
│
├── checkpoints/            # Model weights (auto-downloaded)
├── sam2_repo/              # SAM2 source (auto-cloned)
│
├── client_examples/        # Python client library
│   └── inference_client.py
│
├── scripts/                # Utility scripts
│   ├── download_sam2.sh
│   └── export_sam2_to_onnx.py
│
└── test/                   # Test files
    ├── images/             # Test images
    └── output/             # Generated results
```

## Available Commands

```bash
pixi install              # Install dependencies
pixi run setup           # Complete model setup
pixi run test-sam2       # Run inference test

pixi run download-tiny   # Download tiny model
pixi run download-small  # Download small model
pixi run download-base   # Download base_plus model (default)
pixi run download-large  # Download large model

pixi run export-onnx     # Export models to ONNX
```

## Troubleshooting

**Server won't start:**
```bash
docker logs sam2-triton-server
```

**Model errors:**
- Check `model_repository/sam2_encoder/1/model.onnx` exists
- Check `model_repository/sam2_decoder/1/model.onnx` exists
- Rerun `pixi run export-onnx`

**GPU not detected:**
- Check NVIDIA drivers: `nvidia-smi`
- Check Docker GPU support: `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi`

## Usage

```python
from client_examples.inference_client import SAM2TritonClient

# Connect and encode image
client = SAM2TritonClient()
client.set_image("path/to/image.jpg")

# Segment with point prompt
masks, iou = client.predict(
    point_coords=[[x, y]],  # Click coordinates
    point_labels=[1]         # 1=foreground, 0=background
)

# masks contains logits - threshold at 0 for binary mask
binary_mask = (masks[0, 0] > 0).astype(np.uint8)
```

## Performance

- **Encoder**: ~300ms per image (run once, cached)
- **Decoder**: ~15ms per mask (interactive speed)
- **IoU**: Typically 0.98+ for well-defined objects

See CLAUDE.md for detailed architecture documentation.
