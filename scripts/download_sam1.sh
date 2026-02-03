#!/bin/bash
# Download SAM1 model checkpoints

set -e

CHECKPOINT_DIR="checkpoints"
mkdir -p "$CHECKPOINT_DIR"

# SAM1 model variants and their URLs
declare -A MODELS=(
    ["sam_vit_h_4b8939.pth"]="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    ["sam_vit_l_0b3195.pth"]="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    ["sam_vit_b_01ec64.pth"]="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
)

# Model size to file mapping
declare -A SIZE_TO_FILE=(
    ["vit_h"]="sam_vit_h_4b8939.pth"
    ["vit_l"]="sam_vit_l_0b3195.pth"
    ["vit_b"]="sam_vit_b_01ec64.pth"
)

# Default to vit_h if no argument provided
MODEL_SIZE="${1:-vit_h}"
MODEL_FILE="${SIZE_TO_FILE[$MODEL_SIZE]}"

if [[ -z "$MODEL_FILE" ]]; then
    echo "Error: Invalid model size. Choose from: vit_h, vit_l, vit_b"
    exit 1
fi

MODEL_URL="${MODELS[$MODEL_FILE]}"
OUTPUT_PATH="$CHECKPOINT_DIR/$MODEL_FILE"

if [ -f "$OUTPUT_PATH" ]; then
    echo "Checkpoint already exists: $OUTPUT_PATH"
    exit 0
fi

echo "Downloading SAM1 $MODEL_SIZE ($MODEL_FILE)..."
wget -O "$OUTPUT_PATH" "$MODEL_URL"

echo "Download complete: $OUTPUT_PATH"
