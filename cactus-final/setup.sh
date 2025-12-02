#!/bin/bash
# Setup script for Mac (ARM)

set -e

echo "=========================================="
echo "🌵 Cactus Profile Generation Setup"
echo "=========================================="

# Check architecture
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo "⚠️  Warning: This script is designed for ARM64 (Mac M-series)"
    echo "   Current architecture: $ARCH"
    echo "   You can still install Python dependencies for mock mode."
    echo ""
fi

# Check if we're on Mac
if [[ "$(uname)" != "Darwin" ]]; then
    echo "⚠️  Warning: This script is designed for macOS"
    echo "   Current OS: $(uname)"
    echo ""
fi

# 1. Install Python dependencies
echo ""
echo "📦 Step 1: Installing Python dependencies..."
pip install -r requirements.txt

# 2. Check if Cactus library exists
echo ""
echo "🔍 Step 2: Checking for Cactus library..."

CACTUS_DIR="../../cactus"
LIB_PATH="$CACTUS_DIR/build/cactus/libcactus.dylib"

if [ -f "$LIB_PATH" ]; then
    echo "✅ Found Cactus library: $LIB_PATH"
else
    echo "❌ Cactus library not found"
    echo ""
    echo "Building Cactus library..."

    if [ ! -d "$CACTUS_DIR" ]; then
        echo "❌ Cactus directory not found: $CACTUS_DIR"
        echo "   Please ensure cactus is cloned at the correct location"
        exit 1
    fi

    cd "$CACTUS_DIR"
    ./apple/build.sh
    cd -

    if [ -f "$LIB_PATH" ]; then
        echo "✅ Cactus library built successfully"
    else
        echo "❌ Failed to build Cactus library"
        exit 1
    fi
fi

# 3. Test bindings
echo ""
echo "🧪 Step 3: Testing Cactus bindings..."
python bindings/test_bindings.py

# 4. Check for models
echo ""
echo "📥 Step 4: Checking for embedding models..."

MODELS_DIR="$CACTUS_DIR/weights"
if [ -d "$MODELS_DIR/LFM2-350M" ]; then
    echo "✅ Found LFM2-350M model"
elif [ -d "$MODELS_DIR/Qwen3-0.6B" ]; then
    echo "✅ Found Qwen3-0.6B model"
else
    echo "⚠️  No embedding models found"
    echo ""
    echo "To download embedding model:"
    echo "  cd $CACTUS_DIR"
    echo "  ./cli/cactus download LiquidAI/LFM2-350M"
    echo ""
fi

# Done
echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Download embedding model (if not already done):"
echo "     cd $CACTUS_DIR"
echo "     ./cli/cactus download LiquidAI/LFM2-350M"
echo ""
echo "  2. Generate production profile:"
echo "     python training/generate_profile.py \\"
echo "       --use-cactus \\"
echo "       --model-path $CACTUS_DIR/weights/LFM2-350M/model.gguf \\"
echo "       --output profiles/production_profile.json"
echo ""
