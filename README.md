# SAM + NVIDIA Triton Inference Server

Production-grade deployment of Meta's Segment Anything Models (SAM1, SAM2, SAM3) using NVIDIA Triton Inference Server.

## Quick Links

- **[CLAUDE.md](CLAUDE.md)** - Comprehensive architecture and deployment details

## Features

### Multi-Model Support
- **SAM1** - Original Segment Anything Model (ViT-H, ViT-L, ViT-B)
- **SAM2.1** - 40% faster inference with Hiera backbone (tiny, small, base_plus, large)
- **SAM3** - SAM3 Tracker with multi-scale embeddings

### Triton Benefits
- **Enterprise-grade**: Industry-standard inference protocol
- **Performance**: GPU-accelerated ONNX Runtime with dynamic batching support
- **Scalability**: Native multi-GPU support with load balancing
- **Observability**: Built-in Prometheus metrics
- **Flexibility**: Hot-reload models without downtime

## Quick Start

### Prerequisites
- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- [Pixi](https://pixi.sh) - Modern Python package manager

### Verify GPU Access

```bash
# Test Docker can access GPUs
docker run --rm --gpus all nvcr.io/nvidia/tritonserver:25.01-py3 nvidia-smi
```

### Setup Options

```bash
# 1. Install Pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# 2. Install dependencies
pixi install

# 3. Choose your setup:

# Option A: Setup ALL models (SAM1 + SAM2 + SAM3) - ~5GB download
pixi run setup-all

# Option B: Setup only SAM2 (recommended for quick start)
pixi run setup-sam2

# Option C: Setup individual models
pixi run setup-sam1    # SAM1 only (~2.5GB)
pixi run setup-sam2    # SAM2 only (~350MB)
pixi run setup-sam3    # SAM3 only (~1GB, pre-exported ONNX)

# 4. Start Triton server
docker compose up -d

# 5. Verify deployment
curl http://localhost:8000/v2/models
```

### Test Inference

```bash
# Test SAM2
pixi run test-sam2

# Test SAM3
pixi run test-sam3

# Speculative request stress test
pixi run test-speculative
```

## Model Comparison

| Model | Input Size | Embedding Shape | Best For |
|-------|------------|-----------------|----------|
| **SAM1** | 1024x1024 | (1, 256, 64, 64) | Legacy compatibility, proven accuracy |
| **SAM2** | 1024x1024 | (1, 256, 64, 64) | Production default, video support |
| **SAM3** | 1008x1008 | 3 multi-scale | Latest features, text prompts |

### SAM2 Model Sizes

| Model | Parameters | Memory | Speed | Use Case |
|-------|------------|--------|-------|----------|
| tiny | 39M | 2GB | 91 FPS | Edge devices, real-time |
| small | 46M | 2.5GB | 85 FPS | Balanced |
| **base_plus** | 81M | 4GB | 64 FPS | **Production default** |
| large | 224M | 8GB | 40 FPS | Maximum quality |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       NVIDIA Triton Server                               │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │  SAM1 Encoder   │  │  SAM2 Encoder   │  │  SAM3 Encoder   │          │
│  │  (1024x1024)    │  │  (1024x1024)    │  │  (1008x1008)    │          │
│  │  → (256,64,64)  │  │  → (256,64,64)  │  │  → 3 embeddings │          │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘          │
│           │                    │                    │                    │
│  ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐          │
│  │  SAM1 Decoder   │  │  SAM2 Decoder   │  │  SAM3 Decoder   │          │
│  │  + prompts      │  │  + prompts      │  │  + prompts      │          │
│  │  → masks        │  │  → masks        │  │  → masks        │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
│                                                                          │
│  Ports: HTTP (8000) | gRPC (8001) | Metrics (8002)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Two-Stage Workflow

All SAM models use a two-stage inference pipeline optimized for interactive segmentation:

### Stage 1: Image Encoding (Expensive)
- Run **once** per image
- Generates reusable embeddings
- ~200-800ms depending on model

### Stage 2: Mask Prediction (Fast)
- Run **many times** per image
- Uses cached embeddings + point prompts
- ~10-30ms per prediction

## Project Structure

```
triton-sam2/
├── README.md                 # This file
├── CLAUDE.md                 # Detailed architecture docs
├── docker-compose.yml        # Triton server deployment
├── pyproject.toml            # Pixi tasks and dependencies
│
├── triton_sam/               # Python client module
│   ├── client.py             # SAM2TritonClient (sync)
│   ├── speculative_client.py # Async client with cancellation
│   └── tests/
│
├── scripts/
│   ├── download_sam1.sh      # Download SAM1 checkpoints
│   ├── download_sam2.sh      # Download SAM2 checkpoints
│   ├── download_sam3.sh      # Download SAM3 checkpoints
│   ├── download_sam3_onnx.py # Download pre-exported SAM3 ONNX
│   ├── export_sam1_to_onnx.py # Export SAM1 to ONNX
│   ├── export_sam2_to_onnx.py # Export SAM2 to ONNX
│   └── export_sam3_to_onnx.py # Export SAM3 to ONNX
│
├── model_repository/         # Triton model repository
│   ├── sam1_encoder/
│   │   ├── 1/model.onnx
│   │   └── config.pbtxt
│   ├── sam1_decoder/
│   │   ├── 1/model.onnx
│   │   └── config.pbtxt
│   ├── sam2_encoder/
│   │   ├── 1/model.onnx
│   │   └── config.pbtxt
│   ├── sam2_decoder/
│   │   ├── 1/model.onnx
│   │   └── config.pbtxt
│   ├── sam3_encoder/
│   │   ├── 1/vision_encoder.onnx
│   │   └── config.pbtxt
│   └── sam3_decoder/
│       ├── 1/prompt_encoder_mask_decoder.onnx
│       └── config.pbtxt
│
├── checkpoints/              # Downloaded model weights
├── sam1_repo/                # Cloned segment-anything repo
└── sam2_repo/                # Cloned segment-anything-2 repo
```

## Pixi Tasks Reference

### Setup Tasks

```bash
# Complete setup (all models)
pixi run setup-all      # SAM1 + SAM2 + SAM3

# Individual model setup
pixi run setup-sam1     # Download, clone repo, export SAM1
pixi run setup-sam2     # Download, clone repo, export SAM2 (alias: setup)
pixi run setup-sam3     # Download pre-exported SAM3 ONNX
```

### Download Tasks

```bash
# SAM1 checkpoints
pixi run download-sam1-h    # ViT-Huge (2.5GB, recommended)
pixi run download-sam1-l    # ViT-Large
pixi run download-sam1-b    # ViT-Base

# SAM2 checkpoints
pixi run download-tiny      # 38.9M params
pixi run download-small     # 46M params
pixi run download-base      # 80.8M params (recommended)
pixi run download-large     # 224.4M params

# SAM3 ONNX models
pixi run download-sam3-onnx # Pre-exported from HuggingFace
```

### Export Tasks

```bash
pixi run export-sam1    # Export SAM1 to ONNX
pixi run export-sam2    # Export SAM2 to ONNX
pixi run export-onnx    # Alias for export-sam2
```

### Test Tasks

```bash
pixi run test-sam2          # Basic SAM2 inference test
pixi run test-sam3          # SAM3 inference test
pixi run test-speculative   # Stress test with cancellation
```

### Benchmark Tasks

```bash
pixi run benchmark-sam2     # SAM2 performance benchmark
pixi run benchmark-sam3     # SAM3 performance benchmark
```

## Python Client Usage

### Basic Client (SAM2TritonClient)

```python
from triton_sam import SAM2TritonClient

# Initialize client (supports sam2 or sam3)
client = SAM2TritonClient("localhost:8000", model_type="sam2")

# Encode image once (cached)
client.set_image("image.jpg")

# Predict masks from point prompts
masks, iou = client.predict(
    point_coords=[[512, 512]],  # (x, y) in original image space
    point_labels=[1]             # 1=foreground, 0=background
)

# Threshold logits at 0 for binary mask
binary_mask = (masks[0, 0] > 0).astype(np.uint8)
```

### Speculative Client (Async with Cancellation)

```python
from triton_sam import SpeculativeSAM2Client, queue_multiple_requests
import asyncio

async def interactive_segmentation():
    client = SpeculativeSAM2Client("localhost:8000")
    client.set_image("image.jpg")

    session_id = "user_session_1"

    # Queue many requests (simulating mouse movement)
    coords_list = [np.array([[x, y]]) for x, y in mouse_positions]
    labels_list = [np.array([1]) for _ in mouse_positions]

    tasks = await queue_multiple_requests(
        client, coords_list, labels_list, session_id
    )

    # Cancel intermediate requests when user stops
    client.cancel_session_requests(session_id)

    # Get final result
    result = await wait_for_latest_result(tasks, client, session_id)
```

## Integration with SAM Service API

This Triton deployment integrates with the SAM Service FastAPI application. See the `SAM_service/` directory for the API that provides:

- `POST /embedded_model` - Generate embeddings (supports model_version param)
- `POST /from_model` - Generate masks from embeddings
- `GET /models` - List available models
- `GET /health` - Health check

## Monitoring

```bash
# Server health
curl http://localhost:8000/v2/health/ready

# List models
curl http://localhost:8000/v2/models

# Prometheus metrics
curl http://localhost:8002/metrics | grep nv_inference

# Docker logs
docker compose logs -f
```

## Troubleshooting

### GPU Not Detected
```bash
# Verify NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Models Not Loading
```bash
# Check model files exist
ls -la model_repository/*/1/

# Check Triton logs
docker compose logs triton | grep -i error
```

### Out of Memory
- Use smaller model (tiny or small for SAM2)
- Reduce instance count in config.pbtxt
- Check other GPU processes: `nvidia-smi`

## Performance Expectations

### Latency (base_plus on modern GPU)
- Encoder: ~300ms per image
- Decoder: ~15ms per mask
- End-to-end (1 image, 10 masks): ~450ms

### Memory Requirements
- SAM1 ViT-H: ~4GB
- SAM2 base_plus: ~4GB
- SAM3: ~6GB

## License

Released under the Janelia Open-Source Software License.

## References

- [SAM Paper](https://arxiv.org/abs/2304.02643) - Original Segment Anything
- [SAM2 Paper](https://arxiv.org/abs/2408.00714) - Segment Anything 2
- [SAM2 GitHub](https://github.com/facebookresearch/segment-anything-2)
- [NVIDIA Triton](https://docs.nvidia.com/deeplearning/triton-inference-server/)
