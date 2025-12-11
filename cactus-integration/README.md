# Cactus Integration (ARM/Graviton)

Train router profiles with real Cactus embeddings on ARM (AWS Graviton or local Apple Silicon).

## Prereqs
- ARM Linux (Ubuntu 24.04 recommended) or macOS ARM
- Python 3.10+, build essentials (cmake, clang), git
- Cactus repo in `~/cactus` (cloned by setup script)

## Fast path on AWS Graviton
```bash
# Launch t4g.medium (Ubuntu 24.04), SSH in, then:
git clone https://github.com/AuroraAI/auroraai-router.git
cd auroraai-router/cactus-integration

# Install deps, clone/build Cactus, set LD paths
chmod +x aws/setup_instance.sh && ./aws/setup_instance.sh

# Download embedding model (LFM2-350M, quantized)
chmod +x scripts/download_model.sh && ./scripts/download_model.sh

# Quick embedding check
python3 scripts/test_embeddings.py

# Train profile (adjust samples/output as needed)
python3 scripts/train_profile.py \
  --model models/lfm2-350m-q8.gguf \
  --output profiles/production_profile.json \
  --n-samples 5000
```

Copy the profile back to your machine:
```bash
scp -i <key.pem> ubuntu@<ip>:~/auroraai-router/cactus-integration/profiles/production_profile.json .
```

## Validation (recommended)
- Quick structure/cluster sanity: `python3 scripts/quick_validate.py --profile profiles/production_profile.json`
- HF vs Cactus embedding match (needs GGUF model + built libcactus):  
  `python3 scripts/validate_embedding_consistency.py --profile profiles/production_profile.json --cactus-model models/lfm2-350m-q8.gguf`
- If HDBSCAN shows high noise, rerun `train_profile.py` with KMeans (see script help) or tune cluster params.

## Layout
- `aws/` — instance setup + shared library build scripts
- `scripts/` — Cactus FFI wrapper, training, validation, model download
- `models/` — place GGUF models
- `profiles/` — generated router profiles

## Troubleshooting
- Library errors: rebuild via `aws/build_shared_lib.sh` and export `LD_LIBRARY_PATH=~/cactus/build/cactus:$LD_LIBRARY_PATH`.
- Slow/low-memory: drop `--n-samples` or use a larger instance (t4g.large).
- Embedding mismatch: rerun validation; ensure training and runtime use the same GGUF and preprocessing.
