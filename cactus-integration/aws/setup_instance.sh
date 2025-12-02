#!/bin/bash
# =============================================================================
# AWS Graviton Instance Setup Script
# Run this on a fresh Ubuntu 22.04 ARM64 instance
# =============================================================================

set -e  # Exit on error

echo "=========================================="
echo "🌵 Cactus Integration - AWS Setup"
echo "=========================================="

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install build dependencies
echo "🔧 Installing build dependencies..."
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    clang \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    curl \
    unzip

# Create working directory
echo "📁 Creating working directory..."
mkdir -p ~/cactus-integration
cd ~/cactus-integration

# Clone Cactus repository
echo "📥 Cloning Cactus repository..."
if [ ! -d "cactus" ]; then
    git clone https://github.com/cactus-compute/cactus.git
else
    echo "   Cactus already cloned, pulling latest..."
    cd cactus && git pull && cd ..
fi

# Build Cactus
echo "🔨 Building Cactus library..."
cd cactus/cactus
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ~/cactus-integration

echo "✅ Cactus library built!"
echo "   Location: ~/cactus-integration/cactus/cactus/build/lib/libcactus.a"

# Create Python virtual environment
echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install \
    numpy \
    pandas \
    scikit-learn \
    hdbscan \
    datasets \
    tqdm \
    matplotlib

echo "✅ Python environment ready!"

# Download embedding model weights
echo "📥 Downloading embedding model..."
mkdir -p ~/cactus-integration/weights

# Download LFM2-350M (has embedding support, small size)
cd ~/cactus-integration/cactus
if [ ! -d "weights/lfm2-350m" ]; then
    echo "   Downloading LFM2-350M model..."
    # Use Cactus CLI to download weights
    bash cli/cactus download LiquidAI/LFM2-350M
else
    echo "   Model already downloaded"
fi

cd ~/cactus-integration

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. cd ~/cactus-integration"
echo "  2. source venv/bin/activate"
echo "  3. python train_profile.py"
echo ""
