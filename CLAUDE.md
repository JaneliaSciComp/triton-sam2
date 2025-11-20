# Triton SAM - Segment Anything Model 2 Service

## Overview

This repository provides GPU-accelerated image segmentation using Meta's Segment Anything Model 2 (SAM2), deployed on NVIDIA Triton Inference Server for production-grade performance and scalability.

**Dependency Management**: This project uses [Pixi](https://pixi.sh) for reproducible Python environments and task automation. See the [Setup](#setup-and-deployment) section for installation instructions.

## Architecture

This implementation uses:
- **SAM2.1** - Meta's latest segmentation model with improved accuracy and efficiency
- **NVIDIA Triton** - Enterprise-grade inference server with dynamic batching and multi-GPU support
- **ONNX Runtime** - Optimized model execution for GPU acceleration

## SAM2 + Triton Architecture

### Triton Deployment

#### Model Repository Structure

```
model_repository/
├── sam2_encoder/
│   ├── 1/
│   │   └── model.onnx          # Image encoder ONNX model
│   └── config.pbtxt             # Triton configuration
└── sam2_decoder/
    ├── 1/
    │   └── model.onnx          # Mask decoder ONNX model
    └── config.pbtxt             # Triton configuration
```

#### Two-Stage Inference Pipeline

**Stage 1: Image Encoding (Expensive)**
```
Input: RGB Image (3, 1024, 1024)
   ↓
SAM2 Encoder (Hiera backbone)
   ↓
Output: Image Embeddings (256, 64, 64)
```
- Runs once per image
- ~200-800ms depending on model size and GPU
- Results cached for multiple mask predictions

**Stage 2: Mask Decoding (Fast)**
```
Inputs:
  - Image Embeddings (256, 64, 64)
  - Point Coordinates (N, 2)
  - Point Labels (N,)
   ↓
SAM2 Decoder (Prompt encoder + Mask decoder)
   ↓
Outputs:
  - Segmentation Masks (1, 256, 256)
  - IoU Predictions (1,)
```
- Runs many times per image with different prompts
- ~10-30ms per prediction
- Supports interactive workflows

#### Triton Configuration

**ONNX Runtime with GPU Acceleration**:
- Models run using ONNX Runtime backend on GPU
- Native CUDA acceleration without TensorRT conversion
- TensorRT was initially attempted but SAM2's ONNX operators (e.g., ScatterND) are not fully compatible
- ONNX Runtime provides excellent performance for SAM2's architecture

**Dynamic Input Shapes**:
```protobuf
input [
  {
    name: "image"
    dims: [ -1, 3, 1024, 1024 ]  # -1 = dynamic batch dimension
  }
]
```
- All models configured with dynamic batch dimensions
- Supports variable batch sizes at runtime
- Variable number of prompt points (decoder)

**GPU Instance Configuration**:
```protobuf
instance_group [
  {
    count: 1              # One instance per model
    kind: KIND_GPU
  }
]
```
- Single instance per model for predictable performance
- Can be scaled up for higher throughput
- Multi-GPU support available via instance configuration

### Client Integration

#### Python Client Example

```python
import tritonclient.http as httpclient
import numpy as np

# Connect to Triton
client = httpclient.InferenceServerClient(url="localhost:8000")

# 1. Encode image (once)
image_tensor = preprocess_image("image.jpg")
encoder_input = httpclient.InferInput("image", image_tensor.shape, "FP32")
encoder_input.set_data_from_numpy(image_tensor)

response = client.infer("sam2_encoder", [encoder_input])
embeddings = response.as_numpy("image_embeddings")

# 2. Decode with prompts (many times)
point_coords = np.array([[512, 512]], dtype=np.float32)
point_labels = np.array([1], dtype=np.float32)

decoder_inputs = [
    httpclient.InferInput("image_embeddings", embeddings.shape, "FP32"),
    httpclient.InferInput("point_coords", point_coords.shape, "FP32"),
    httpclient.InferInput("point_labels", point_labels.shape, "FP32")
]

response = client.infer("sam2_decoder", decoder_inputs)
masks = response.as_numpy("masks")
iou = response.as_numpy("iou_predictions")
```

See `client_examples/inference_client.py` for a complete implementation.

#### HTTP API Example

Triton provides a REST API for language-agnostic integration:

```bash
# Check server health
curl http://localhost:8000/v2/health/ready

# List models
curl http://localhost:8000/v2/models

# Model metadata
curl http://localhost:8000/v2/models/sam2_encoder

# Inference (requires binary payload)
curl -X POST http://localhost:8000/v2/models/sam2_encoder/infer \
  -H "Content-Type: application/json" \
  -d @inference_request.json
```

### Setup and Deployment

#### Prerequisites

- **Pixi**: Modern Python package manager ([installation](https://pixi.sh))
  ```bash
  curl -fsSL https://pixi.sh/install.sh | bash
  ```
- **Docker**: With NVIDIA Container Toolkit for GPU support
- **NVIDIA GPU**: With updated drivers (CUDA 12.x recommended)

#### Quick Start

1. **Install dependencies and setup environment**:
```bash
pixi install
```

This creates a reproducible environment with all dependencies including PyTorch, ONNX Runtime, and SAM2.

2. **Complete setup (download model, clone SAM2 repo, export to ONNX)**:
```bash
pixi run setup
```

This single command:
- Downloads the SAM2.1 base_plus checkpoint (~350MB)
- Clones the SAM2 repository for model configs
- Exports encoder and decoder to ONNX format

**Alternative: Manual steps**
```bash
pixi run download-base       # Download model checkpoint
pixi run clone-sam2          # Clone SAM2 repository
pixi run export-onnx         # Export to ONNX format
```

3. **Start Triton server**:
```bash
docker compose up -d
```

4. **Test inference**:
```bash
pixi run python client_examples/inference_client.py \
    --image test_image.jpg \
    --points 512,512 \
    --output mask.png \
    --visualize
```

#### Available Pixi Tasks

```bash
pixi run download-tiny       # Download tiny model (38.9M params)
pixi run download-small      # Download small model (46M params)
pixi run download-base       # Download base_plus model (80.8M params, recommended)
pixi run download-large      # Download large model (224.4M params)

pixi run setup               # Complete setup: download + clone + export
pixi run export-onnx         # Export models to ONNX
pixi run test-sam2           # Run inference test
pixi run format              # Format code with black
pixi run lint                # Lint code with ruff
```

#### Docker Deployment

The `docker-compose.yml` provides a production-ready setup:
- Automatic GPU allocation
- Shared memory configuration for optimal performance
- Model repository mounted as volume
- HTTP (8000), gRPC (8001), and Metrics (8002) endpoints

**Troubleshooting Snap Docker**:

If Docker is installed via Snap and you get "no configuration file provided: not found" errors:

1. Grant removable-media permission:
```bash
sudo snap connect docker:removable-media
sudo systemctl restart snap.docker.dockerd
```

2. If the project is in a non-standard mount (e.g., `/groups/`), Snap Docker may still not have access. Copy the project to your home directory:
```bash
cp -r /path/to/triton_sam ~/triton_sam_work
cd ~/triton_sam_work
sudo docker-compose up -d
```

3. Note: You'll need to use `sudo` for all docker commands with Snap Docker unless you install Docker via traditional packages.

4. The warning `the attribute 'version' is obsolete` can be safely ignored - it's just informing you that the version field in docker-compose.yml is deprecated in newer versions.

#### Monitoring

Triton exposes Prometheus-compatible metrics:
```bash
curl http://localhost:8002/metrics
```

Key metrics:
- `nv_inference_request_success`: Request count by model
- `nv_inference_request_duration_us`: Latency distribution
- `nv_gpu_utilization`: GPU utilization percentage
- `nv_inference_queue_duration_us`: Time spent in queue

### Performance Characteristics

**Expected Latency** (base_plus model):
- Encoder: ~300ms per image (varies by GPU)
- Decoder: ~15ms per mask
- End-to-end (1 image, 1 prompt): ~315ms
- End-to-end (1 image, 10 prompts): ~450ms

**GPU Compatibility**:
- Tested on Blackwell architecture (RTX PRO 6000)
- Requires Triton 25.01+ for Blackwell support
- Compatible with Ampere (A100), Hopper, and newer architectures
- Requires CUDA 12.x drivers

**Throughput**:
- Single-request latency optimized (no batching by default)
- Can enable dynamic batching for higher throughput use cases
- Performance scales with GPU compute capability

**Memory Requirements**:
- Tiny: ~2GB GPU memory
- Small: ~2.5GB GPU memory
- Base Plus: ~4GB GPU memory
- Large: ~8GB GPU memory

---

