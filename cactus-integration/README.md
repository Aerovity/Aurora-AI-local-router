# Cactus Integration for AuroraAI Router Training

This package allows you to train AuroraAI Router profiles using **real Cactus embeddings** on AWS Graviton (ARM64) instances.

## Why This Exists

The AuroraAI Router uses k-means cluster centers to route queries to appropriate models. For routing to work correctly at runtime, the cluster centers must be generated using the **same embedding function** that will be used on mobile devices.

- **HuggingFace embeddings**: FP32 precision, x86/ARM
- **Cactus embeddings**: FP16/INT8 quantized, ARM-only, optimized for mobile

Using HuggingFace to generate cluster centers would result in mismatched embeddings at runtime, causing incorrect routing decisions.

## Requirements

- **AWS Account** with ~$5 credits (training takes ~1-2 hours)
- Cactus-compatible embedding model (LFM2-350M recommended)

## Quick Start

### 1. Launch AWS Graviton Instance

1. Go to AWS Console → EC2 → Launch Instance
2. Select **Ubuntu 24.04 LTS (ARM64)**
3. Instance type: **t4g.medium** (~$0.03/hr, 2 vCPU, 4GB RAM)
4. Storage: 30GB gp3
5. Security group: Allow SSH (port 22)
6. Launch and download key pair

### 2. Connect and Setup

```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@<instance-ip>

# Clone this repo (or upload via scp)
git clone https://github.com/AuroraAI/auroraai-router.git
cd auroraai-router/cactus-integration

# Run setup (installs deps, clones Cactus, builds shared lib)
chmod +x aws/setup_instance.sh
./aws/setup_instance.sh
```

### 3. Download Embedding Model

```bash
# Download LFM2-350M in GGUF format
chmod +x scripts/download_model.sh
./scripts/download_model.sh
```

### 4. Test Embeddings

```bash
# Quick test to verify Cactus embeddings work
python3 scripts/test_embeddings.py
```

### 5. Train Profile

```bash
# Full training run (takes 30-60 minutes)
python3 scripts/train_profile.py \
    --model models/lfm2-350m-q8.gguf \
    --output profiles/production_profile.json \
    --n-samples 5000

# For quick test run
python3 scripts/train_profile.py \
    --model models/lfm2-350m-q8.gguf \
    --output profiles/test_profile.json \
    --n-samples 500
```

### 6. Download Profile

```bash
# On your local machine
scp -i your-key.pem ubuntu@<instance-ip>:~/auroraai-router/cactus-integration/profiles/production_profile.json .
```

### 7. Terminate Instance

Don't forget to terminate your EC2 instance to stop charges!

## Profile Format

The generated profile is a JSON file compatible with the AuroraAI Router SDK:

```json
{
  "version": "2.0",
  "embedding_model": "lfm2-350m-q8",
  "embedding_dim": 1024,
  "n_clusters": 12,
  "cluster_centers": [[...], [...], ...],
  "cluster_labels": ["reasoning", "creative", ...],
  "cluster_stats": {...},
  "training_info": {...}
}
```

## Files

```
cactus-integration/
├── README.md                    # This file
├── aws/
│   ├── setup_instance.sh        # Ubuntu setup script
│   └── build_shared_lib.sh      # Builds libcactus.so
├── scripts/
│   ├── cactus_wrapper.py        # Python bindings for Cactus FFI
│   ├── train_profile.py         # Main training script
│   ├── test_embeddings.py       # Verify embeddings work
│   └── download_model.sh        # Download embedding model
├── models/                      # GGUF models go here
└── profiles/                    # Generated profiles
```

## Troubleshooting

### "Library not found" error
```bash
# Rebuild shared library
cd ~/cactus && ./aws/build_shared_lib.sh
export LD_LIBRARY_PATH=~/cactus/build/cactus:$LD_LIBRARY_PATH
```

### Out of memory
Use a larger instance (t4g.large = 8GB RAM) or reduce batch size:
```bash
python3 scripts/train_profile.py --batch-size 16
```

### Slow embedding generation
This is expected - Cactus on CPU is slower than GPU. The t4g.medium processes ~10-20 embeddings/second.

## Cost Estimate

| Instance | vCPU | RAM | Cost/hr | 5000 samples |
|----------|------|-----|---------|--------------|
| t4g.micro | 2 | 1GB | $0.008 | ~$0.10 |
| t4g.medium | 2 | 4GB | $0.034 | ~$0.50 |
| t4g.large | 2 | 8GB | $0.067 | ~$1.00 |

Recommended: t4g.medium for ~$0.50 total cost.

## License

MIT License - Same as AuroraAI Router
