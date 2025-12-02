#!/bin/bash
# Download embedding model for Cactus
# Run this after setup_instance.sh

set -e

MODEL_DIR="$(dirname "$0")/../models"
mkdir -p "$MODEL_DIR"

echo "=========================================="
echo "  Download Embedding Model for Cactus"
echo "=========================================="

# LFM2-350M is recommended for embeddings
# It's small (~350MB) and has good quality

MODEL_URL="https://huggingface.co/cactus-ai/lfm2-350m-gguf/resolve/main/lfm2-350m-q8_0.gguf"
MODEL_FILE="$MODEL_DIR/lfm2-350m-q8.gguf"

if [ -f "$MODEL_FILE" ]; then
    echo "Model already exists: $MODEL_FILE"
    exit 0
fi

echo "Downloading LFM2-350M (Q8_0 quantization)..."
echo "This may take a few minutes depending on your connection."
echo ""

# Check if wget is available, otherwise use curl
if command -v wget &> /dev/null; then
    wget -O "$MODEL_FILE" "$MODEL_URL" --progress=bar:force
elif command -v curl &> /dev/null; then
    curl -L -o "$MODEL_FILE" "$MODEL_URL" --progress-bar
else
    echo "ERROR: Neither wget nor curl found. Please install one."
    exit 1
fi

echo ""
echo "Download complete!"
echo "Model saved to: $MODEL_FILE"
echo ""

# Verify file size (should be ~350MB)
FILE_SIZE=$(stat -f%z "$MODEL_FILE" 2>/dev/null || stat -c%s "$MODEL_FILE" 2>/dev/null)
if [ "$FILE_SIZE" -lt 100000000 ]; then
    echo "WARNING: File seems too small. Download may have failed."
    echo "Try downloading manually from:"
    echo "  $MODEL_URL"
fi

echo "Ready to train! Run:"
echo "  python3 scripts/train_profile.py --model $MODEL_FILE"
