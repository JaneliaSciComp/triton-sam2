#!/bin/bash
# Download SAM2 model checkpoints

set -e

CHECKPOINT_DIR="checkpoints"
mkdir -p "$CHECKPOINT_DIR"

# SAM2.1 model variants and their URLs
declare -A MODELS=(
    ["sam2.1_hiera_tiny.pt"]="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
    ["sam2.1_hiera_small.pt"]="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
    ["sam2.1_hiera_base_plus.pt"]="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
    ["sam2.1_hiera_large.pt"]="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
)

# Default to base_plus if no argument provided
MODEL_SIZE="${1:-base_plus}"
MODEL_FILE="sam2.1_hiera_${MODEL_SIZE}.pt"

if [[ ! -v "MODELS[$MODEL_FILE]" ]]; then
    echo "Error: Invalid model size. Choose from: tiny, small, base_plus, large"
    exit 1
fi

MODEL_URL="${MODELS[$MODEL_FILE]}"
OUTPUT_PATH="$CHECKPOINT_DIR/$MODEL_FILE"

if [ -f "$OUTPUT_PATH" ]; then
    echo "Checkpoint already exists: $OUTPUT_PATH"
    exit 0
fi

echo "Downloading $MODEL_FILE..."
wget -O "$OUTPUT_PATH" "$MODEL_URL"

echo "Download complete: $OUTPUT_PATH"
