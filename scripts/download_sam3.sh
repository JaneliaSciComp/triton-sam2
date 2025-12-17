#!/bin/bash
# Download SAM3 model checkpoints from HuggingFace

set -e

CHECKPOINT_DIR="checkpoints"
mkdir -p "$CHECKPOINT_DIR"

# SAM3 model checkpoint
MODEL_REPO="facebook/sam3"
MODEL_FILE="sam3_hiera_base_plus.pt"
OUTPUT_PATH="$CHECKPOINT_DIR/$MODEL_FILE"

if [ -f "$OUTPUT_PATH" ]; then
    echo "Checkpoint already exists: $OUTPUT_PATH"
    exit 0
fi

echo "=================================================="
echo "Downloading SAM3 checkpoint from HuggingFace"
echo "=================================================="
echo ""
echo "Note: SAM3 checkpoints require HuggingFace authentication."
echo "If you haven't already, please run: huggingface-cli login"
echo ""

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface-hub..."
    python -m pip install -q huggingface-hub[cli]
fi

# Check if user is logged in
if ! huggingface-cli whoami &> /dev/null; then
    echo "Error: Not logged in to HuggingFace."
    echo "Please run: huggingface-cli login"
    echo ""
    echo "You'll need:"
    echo "  1. A HuggingFace account (https://huggingface.co/join)"
    echo "  2. Request access to SAM3 model at: https://huggingface.co/$MODEL_REPO"
    echo "  3. Generate an access token at: https://huggingface.co/settings/tokens"
    exit 1
fi

echo "Downloading $MODEL_FILE from $MODEL_REPO..."
huggingface-cli download "$MODEL_REPO" "$MODEL_FILE" \
    --local-dir "$CHECKPOINT_DIR" \
    --local-dir-use-symlinks False

echo ""
echo "✓ Download complete: $OUTPUT_PATH"
