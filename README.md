# SAM2 + NVIDIA Triton Inference Server

Production-grade deployment of Meta's Segment Anything Model 2 (SAM2) using NVIDIA Triton Inference Server.

## Quick Links

- **[CLAUDE.md](CLAUDE.md)** - Comprehensive architecture and deployment details

## Features

### SAM2.1 Model
- **40% faster** inference than SAM 1.0
- **Better accuracy** with Hiera backbone architecture
- **4 model sizes**: tiny (39M), small (46M), base_plus (81M), large (224M)
- **Video support**: Unified architecture for images and videos

### Triton Benefits
- **Enterprise-grade**: Industry-standard inference protocol
- **Performance**: GPU-accelerated ONNX Runtime with dynamic batching support
- **Scalability**: Native multi-GPU support with load balancing
- **Observability**: Built-in Prometheus metrics
- **Flexibility**: Hot-reload models without downtime

## Quick Start

### Prerequisites
- NVIDIA GPU (compute capability 7.0+, Blackwell architecture supported)
- Docker with NVIDIA Container Toolkit
- [Pixi](https://pixi.sh) - Modern Python package manager

### 3-Minute Setup

```bash
# 1. Install Pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# 2. Install dependencies and create environment
pixi install

# 3. Complete setup (download model, clone SAM2, export to ONNX)
pixi run setup

# 4. Start Triton server
docker compose up -d

# 5. Verify deployment
curl http://localhost:8000/v2/health/ready
```

### Test Inference

```bash
# Run inference with the example client
pixi run python client_examples/inference_client.py \
    --image path/to/image.jpg \
    --points 512,512 \
    --output mask.png \
    --visualize
```

### Manual Setup (Alternative)

If you prefer step-by-step control:

```bash
pixi run download-base       # Download SAM2.1 base_plus model
pixi run clone-sam2          # Clone SAM2 repository
pixi run export-onnx         # Export to ONNX format
docker compose up -d         # Start server
```

## Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│                    NVIDIA Triton Server                    │
│                                                            │
│  ┌──────────────────────┐      ┌──────────────────────┐    │
│  │   SAM2 Encoder       │      │   SAM2 Decoder       │    │
│  │   (ONNX Runtime)     │      │   (ONNX Runtime)     │    │
│  │                      │      │                      │    │
│  │  Input:              │      │  Inputs:             │    │
│  │  - Image (1024x1024) │      │  - Embeddings        │    │
│  │                      │      │  - Point coords      │    │
│  │  Output:             │      │  - Point labels      │    │
│  │  - Embeddings        │      │                      │    │
│  │    (256x64x64)       │      │  Outputs:            │    │
│  │                      │      │  - Masks (256x256)   │    │
│  │  ~300ms              │      │  - IoU scores        │    │
│  │                      │      │                      │    │
│  │                      │      │  ~15ms per mask      │    │
│  └──────────────────────┘      └──────────────────────┘    │
│                                                            │
│  Features: GPU Acceleration, Multi-GPU, Metrics, HTTP/gRPC │
└────────────────────────────────────────────────────────────┘
         ↑                                    ↑
         │                                    │
    HTTP/gRPC API                        Python Client
  (localhost:8000)                   (tritonclient library)
```

## Two-Stage Workflow

SAM2, like SAM 1.0, uses a two-stage inference pipeline optimized for interactive segmentation:

### Stage 1: Image Encoding (Expensive)
- Run **once** per image
- Generates reusable embeddings
- Cache for multiple predictions
- ~200-800ms depending on model size

### Stage 2: Mask Prediction (Fast)
- Run **many times** per image
- Uses cached embeddings + new prompts
- Interactive latency
- ~10-30ms per prediction

This design enables responsive user interfaces where users can click points to segment objects in real-time.

## Model Selection Guide

| Model | Best For | Memory | Speed |
|-------|----------|--------|-------|
| **tiny** | Edge devices, mobile, real-time preview | 2GB | Fastest (91 FPS) |
| **small** | Balanced use cases | 2.5GB | Very fast (85 FPS) |
| **base_plus** | **Production default** (recommended) | 4GB | Fast (64 FPS) |
| **large** | Maximum quality, research | 8GB | Slower (40 FPS) |

## Project Structure

```
triton_sam/
├── CLAUDE.md                 # Detailed architecture documentation
├── README.md                 # This file
├── docker-compose.yml        # Triton server deployment
├── pyproject.toml            # Pixi configuration and Python dependencies
├── test_sam2.py              # Test script with visualization
│
├── scripts/
│   ├── download_sam2.sh      # Download SAM2 checkpoints
│   └── export_sam2_to_onnx.py # Export models to ONNX
│
├── model_repository/         # Triton model repository
│   ├── sam2_encoder/
│   │   ├── 1/
│   │   │   └── model.onnx
│   │   └── config.pbtxt
│   └── sam2_decoder/
│       ├── 1/
│       │   └── model.onnx
│       └── config.pbtxt
│
├── client_examples/
│   └── inference_client.py   # Python client library
│
├── checkpoints/              # Downloaded model weights
├── sam2_repo/                # Cloned SAM2 repository
│
└── test/
    ├── images/               # Test input images
    └── output/               # Generated masks and visualizations
```

## Model Export Process

SAM2 models are converted from PyTorch to ONNX format for deployment on Triton. The export process splits the model into two separate components optimized for different inference patterns:

### Two-Stage Architecture

**Stage 1: Encoder (Expensive)**
- **Input**: RGB image (1, 3, 1024, 1024)
- **Output**: Image embeddings (1, 256, 64, 64)
- **Purpose**: Processes the full image once to generate reusable embeddings
- **Typical latency**: 200-800ms depending on model size
- **Usage pattern**: Run once per image, cache embeddings

**Stage 2: Decoder (Fast)**
- **Inputs**:
  - Image embeddings (1, 256, 64, 64)
  - Point coordinates (B, N, 2) - user click positions
  - Point labels (B, N) - foreground (1) or background (0)
- **Outputs**:
  - Segmentation masks (B, 1, 256, 256) - logits (threshold at 0)
  - IoU predictions (B, 1) - confidence scores
- **Purpose**: Generate masks from prompts using cached embeddings
- **Typical latency**: 10-30ms per mask
- **Usage pattern**: Run many times with different prompts per image

### Export Process Details

The `export_sam2_to_onnx.py` script handles several critical transformations:

#### 1. Model Loading
```python
model = build_sam2(model_cfg, checkpoint, device="cpu")
```
- Uses CPU for export (models will run on GPU in Triton)
- Loads SAM2.1 architecture with Hiera backbone

#### 2. Encoder Export
```python
torch.onnx.export(
    encoder,
    dummy_input,
    output_path,
    opset_version=17,
    dynamic_axes={"image": {0: "batch_size"}}
)
```
- Exports image encoder as standalone model
- Dynamic batch size support for batching requests
- ONNX opset 17 for compatibility with Triton

#### 3. Decoder Export with Fixes
```python
class SAM2DecoderONNX(torch.nn.Module):
    def forward(self, image_embeddings, point_coords, point_labels):
        # Disable high_res_features to avoid unpacking issues
        self.sam_mask_decoder.use_high_res_features = False

        low_res_masks, iou_predictions, _, _ = self.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            high_res_features=None
        )
```

**Key fix**: SAM2.1's `use_high_res_features` flag is temporarily disabled during export to prevent unpacking errors. This feature expects a tuple of high-resolution feature maps that aren't available during ONNX tracing.

#### 4. Dynamic Axes Configuration
```python
dynamic_axes={
    "point_coords": {0: "batch_size", 1: "num_points"},
    "point_labels": {0: "batch_size", 1: "num_points"},
    "masks": {0: "batch_size"}
}
```
- Supports variable batch sizes for dynamic batching
- Supports variable number of prompt points per request

### Why CPU Export?

The models are exported on CPU but run on GPU in Triton because:
1. **CUDA availability**: Pixi environments may not have PyTorch with CUDA
2. **Portability**: CPU export works on any machine
3. **Performance**: Export is a one-time operation; runtime performance is unaffected
4. **Compatibility**: Ensures ONNX operators are compatible across devices

## Scripts Documentation

### download_sam2.sh

Downloads SAM2.1 model checkpoints from Meta's official repository.

**Usage:**
```bash
bash scripts/download_sam2.sh [MODEL_SIZE]
```

**Arguments:**
- `MODEL_SIZE`: One of `tiny`, `small`, `base_plus`, `large` (default: `base_plus`)

**Example:**
```bash
# Download base_plus model (recommended)
bash scripts/download_sam2.sh base_plus

# Download tiny model for edge deployment
bash scripts/download_sam2.sh tiny
```

**Behavior:**
- Downloads checkpoint to `checkpoints/sam2.1_hiera_[SIZE].pt`
- Skips download if checkpoint already exists
- Validates model size argument
- Uses wget for reliable downloads (~150-350MB depending on model)

**Model URLs:**
- Tiny (38.9M): `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt`
- Small (46M): `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt`
- Base Plus (80.8M): `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt`
- Large (224.4M): `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt`

**Pixi Tasks:**
```bash
pixi run download-tiny    # Download tiny model
pixi run download-small   # Download small model
pixi run download-base    # Download base_plus model
pixi run download-large   # Download large model
```

### export_sam2_to_onnx.py

Exports SAM2 PyTorch models to ONNX format for Triton deployment.

**Usage:**
```bash
python scripts/export_sam2_to_onnx.py \
    --checkpoint CHECKPOINT \
    --model-cfg MODEL_CFG \
    [--output-dir OUTPUT_DIR] \
    [--image-size IMAGE_SIZE] \
    [--device DEVICE]
```

**Arguments:**
- `--checkpoint` (required): Path to SAM2 checkpoint file (e.g., `checkpoints/sam2.1_hiera_base_plus.pt`)
- `--model-cfg` (required): Path to SAM2 config YAML (e.g., `sam2_repo/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml`)
- `--output-dir`: Output directory for ONNX models (default: `model_repository`)
- `--image-size`: Input image size (default: `1024`)
- `--device`: Export device (default: `cpu`, can be `cuda` if available)

**Example:**
```bash
python scripts/export_sam2_to_onnx.py \
    --checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
    --model-cfg sam2_repo/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml \
    --output-dir model_repository \
    --device cpu
```

**Output Structure:**
```
model_repository/
├── sam2_encoder/
│   └── 1/
│       └── model.onnx    # Encoder ONNX model (~320MB for base_plus)
└── sam2_decoder/
    └── 1/
        └── model.onnx    # Decoder ONNX model (~180MB for base_plus)
```

**Process:**
1. Loads SAM2 model from checkpoint
2. Exports image encoder with dynamic batch size
3. Exports decoder with wrapper class that:
   - Combines prompt encoder + mask decoder
   - Disables high_res_features for ONNX compatibility
   - Configures dynamic axes for batching
4. Saves models to Triton model repository structure

**Pixi Task:**
```bash
pixi run export-onnx   # Exports base_plus model by default
```

**Important Notes:**
- Uses ONNX opset 17 for Triton compatibility
- Disables `use_high_res_features` to prevent unpacking errors
- Supports dynamic batch sizes for Triton's dynamic batching
- Export warnings about TracerWarning are normal and suppressed

## Performance Expectations

### Latency (SAM2.1 base_plus on A100 GPU)
- Single image encoding: ~300ms
- Single mask prediction: ~15ms
- End-to-end (1 image, 1 mask): ~315ms
- End-to-end (1 image, 10 masks): ~450ms

### Throughput
- Single-request optimized (no batching by default)
- Encoder: ~3-5 images/second per instance
- Decoder: ~60-100 masks/second per instance
- Can enable dynamic batching for higher concurrent throughput


## Monitoring and Debugging

### Check Server Status
```bash
# Server health
curl http://localhost:8000/v2/health/ready

# List loaded models
curl http://localhost:8000/v2/models

# Model-specific status
curl http://localhost:8000/v2/models/sam2_encoder/ready
```

### View Metrics
```bash
# Prometheus metrics
curl http://localhost:8002/metrics

# Filter specific metrics
curl http://localhost:8002/metrics | grep nv_inference
```

### View Logs
```bash
# Docker logs
docker logs sam2-triton-server -f

# With timestamps
docker logs sam2-triton-server -f --timestamps
```

## Common Issues

### Out of Memory
- Use a smaller model (tiny or small)
- Reduce instance count in config.pbtxt
- Reduce batch sizes

### Slow Inference
- Check GPU utilization: `nvidia-smi`
- Verify ONNX Runtime is using GPU (check server logs)
- Ensure CUDA drivers are up to date (12.x recommended)
- Consider using a smaller model (tiny or small) for faster inference

### Model Not Loading
- Verify ONNX files exist in correct paths
- Check config.pbtxt syntax
- Review server logs for errors

## Use Cases

- **Interactive Segmentation**: Paintera, 3D Slicer, medical imaging tools
- **Batch Processing**: Large-scale image annotation pipelines
- **Video Segmentation**: Track objects across video frames
- **Edge Deployment**: Mobile and embedded applications (tiny model)

## Contributing

This is an internal research tool. For issues or improvements:
1. Open an issue describing the problem
2. Include logs and configuration details
3. Test with the latest Triton server version

## License

Released under the Janelia Open-Source Software License.

## References

- [SAM2 Paper](https://arxiv.org/abs/2408.00714)
- [SAM2 GitHub](https://github.com/facebookresearch/segment-anything-2)
- [NVIDIA Triton Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [Original SAM Paper](https://arxiv.org/abs/2304.02643)
