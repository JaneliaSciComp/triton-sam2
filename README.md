# SAM2 + NVIDIA Triton Inference Server

Production-grade deployment of Meta's Segment Anything Model 2 (SAM2) using NVIDIA Triton Inference Server.

## Quick Links

- **[QUICKSTART.md](QUICKSTART.md)** - 3-step installation guide
- **[CLAUDE.md](CLAUDE.md)** - Comprehensive architecture and model details

## Features

### SAM2.1 Model
- **40% faster** inference than SAM 1.0
- **Better accuracy** with Hiera backbone architecture
- **4 model sizes**: tiny (39M), small (46M), base_plus (81M), large (224M)
- **Video support**: Unified architecture for images and videos

### Triton Benefits
- **Enterprise-grade**: Industry-standard inference protocol
- **Performance**: TensorRT optimization + dynamic batching
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
┌─────────────────────────────────────────────────────────────┐
│                     NVIDIA Triton Server                     │
│                                                              │
│  ┌──────────────────────┐      ┌──────────────────────┐    │
│  │   SAM2 Encoder       │      │   SAM2 Decoder       │    │
│  │   (ONNX/TensorRT)    │      │   (ONNX/TensorRT)    │    │
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
│                                                              │
│  Features: Dynamic Batching, Multi-GPU, TensorRT, Metrics   │
└─────────────────────────────────────────────────────────────┘
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
├── SETUP.md                  # Setup and deployment guide
├── README.md                 # This file
├── docker-compose.yml        # Triton server deployment
├── pyproject.toml            # Pixi configuration and Python dependencies
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
│   └── inference_client.py   # Python client example
│
├── checkpoints/              # Downloaded model weights
│
```

## Performance Expectations

### Latency (SAM2.1 base_plus on A100 GPU)
- Single image encoding: ~300ms
- Single mask prediction: ~15ms
- End-to-end (1 image, 1 mask): ~315ms
- End-to-end (1 image, 10 masks): ~450ms

### Throughput (with dynamic batching)
- Encoder: 10-15 images/second
- Decoder: 200-300 masks/second


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
- Enable TensorRT (already configured)
- Use FP16 precision (already configured)
- Check GPU utilization: `nvidia-smi`
- Adjust dynamic batching parameters

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
