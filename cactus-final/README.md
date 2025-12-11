# AuroraAI Router — Cactus Profile Generation

Generate router profiles using the Cactus C library. Supports real Cactus embeddings on ARM/macOS and a mock mode on x86 for structure testing.

## What it does
- Loads ~2k MMLU samples, builds embeddings, clusters (KMeans/HDBSCAN), simulates per-cluster error rates, and writes a profile JSON.
- Two modes: `--use-cactus` (real embeddings) or `--mock-embeddings` (sentence-transformers).
- C library lookup checks the xcframework output at `../cactus/apple/cactus-macos.xcframework/.../cactus` by default; override with `--lib-path` if needed.

## Prereqs
- Python 3.10+, `uv` (or `pip`)
- Cactus repo checked out at `../cactus`
- For real embeddings: run `../cactus/apple/build.sh` and download a model (e.g., `./cli/cactus download LiquidAI/LFM2-350M`, stored in `../cactus/weights/lfm2-350m`)

## Quick start (mock mode, works on x86)
```bash
cd cactus-final
uv sync --extra mock
uv run python training/generate_profile.py \
  --mock-embeddings \
  --output profiles/test_profile.json
```

## Quick start (real Cactus, ARM/macOS)
```bash
cd ../cactus && ./apple/build.sh
./cli/cactus download LiquidAI/LFM2-350M  # saves to weights/lfm2-350m

cd ../aurora-AI-local-router/cactus-final
uv sync
uv run python training/generate_profile.py \
  --use-cactus \
  --model-path ../cactus/weights/lfm2-350m \
  --output profiles/production_profile.json
# Optional: --lib-path ../cactus/apple/cactus-macos.xcframework/macos-arm64/cactus.framework/Versions/A/cactus
```

## Scripts and layout
- `bindings/cactus_bindings.py` — ctypes wrapper; default lib search includes the macOS xcframework output.
- `training/generate_profile.py` — main pipeline, caches embeddings in `data/`.
- `training/config.py` — model list + dataset/config knobs.
- `training/utils.py` — clustering, error-rate simulation, caching helpers.

## Troubleshooting
- “Cactus library not found”: build via `../cactus/apple/build.sh` or pass `--lib-path` to the compiled library inside the xcframework.
- “Model file not found”: `--model-path` must point to the model directory (containing `config.txt`), e.g., `../cactus/weights/lfm2-350m`.
- On x86: use `--mock-embeddings`; real Cactus requires ARM.
