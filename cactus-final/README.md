# 🌵 AuroraAI Router - Cactus Profile Generation

**Clean restart of AuroraAI Router training with Cactus C bindings.**

This project generates router profiles for 12 local Cactus models using **real Cactus embeddings** via C library bindings. It follows the COLAB notebook strategy but uses native Cactus instead of HuggingFace transformers.

## 🎯 What This Does

1. **Loads MMLU dataset** (~2000 samples, 15 diverse topics)
2. **Extracts embeddings** using Cactus C library (`cactus_embed()`)
3. **Clusters embeddings** using KMeans and HDBSCAN
4. **Computes error rates** per cluster per model (simulated for now)
5. **Saves router profile** (~100KB JSON file)

## 🏗️ Architecture

```
┌─────────────────────┐
│  MMLU Dataset       │  15 topics, ~2000 samples
│  (Text Questions)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Cactus C Library   │  libcactus.dylib (Mac) or libcactus.so (Linux ARM)
│  cactus_embed()     │
└──────────┬──────────┘
           │
           ▼  768-dim embeddings (float16)
┌─────────────────────┐
│  Clustering         │  KMeans / HDBSCAN
│  (scikit-learn)     │  Find optimal K by silhouette score
└──────────┬──────────┘
           │
           ▼  Cluster centers + labels
┌─────────────────────┐
│  Error Rate Calc    │  Simulated based on model size
│  (per cluster)      │  TODO: Replace with real Cactus inference
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Router Profile     │  JSON file with:
│  (JSON)             │  - Cluster centers (float16)
│                     │  - Error rates per model
│                     │  - Model metadata
└─────────────────────┘
```

## 📋 Project Structure

```
cactus-final/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
│
├── bindings/                     # Python ↔ C bindings
│   ├── __init__.py
│   ├── cactus_bindings.py       # ctypes wrapper for libcactus
│   └── test_bindings.py         # Test script
│
├── training/                     # Profile generation
│   ├── __init__.py
│   ├── config.py                # 12 Cactus models + config
│   ├── utils.py                 # Helper functions
│   └── generate_profile.py      # Main training script
│
├── data/                         # MMLU cache
│   └── .gitkeep
│
├── profiles/                     # Generated profiles
│   └── .gitkeep
│
└── models/                       # GGUF models (download on Mac)
    └── .gitkeep
```

## 🚀 Quick Start

### On x86 (Windows/Intel Mac) - Mock Mode

You can test the project structure on x86 using HuggingFace embeddings:

```bash
cd auroraai-router/cactus-final

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt

# For mock embeddings support (optional)
uv sync --extra mock

# Test bindings (will show "not available" on x86)
uv run python bindings/test_bindings.py

# Generate profile with mock embeddings (for testing)
uv run python training/generate_profile.py \
    --mock-embeddings \
    --output profiles/test_profile.json
```

**⚠️ Note:** Mock embeddings use `sentence-transformers` and are NOT identical to Cactus embeddings. Use only for testing the pipeline structure.

---

### On Mac (ARM) - Real Cactus Mode

#### 1. Build Cactus Library

```bash
# Navigate to cactus directory
cd ../../cactus

# Build Cactus (generates libcactus.dylib)
./apple/build.sh

# Verify build
ls -lh build/cactus/libcactus.dylib
```

#### 2. Install Python Dependencies

```bash
cd ../auroraai-router/cactus-final

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

#### 3. Download Embedding Model

```bash
cd ../../cactus

# Use Cactus CLI to download embedding model
# Recommended: LFM2-350M (233MB, has embed capability)
./cli/cactus download LiquidAI/LFM2-350M

# Or use Qwen-600M (394MB, also has embed)
# ./cli/cactus download Qwen/Qwen3-0.6B

# Models are saved to: weights/
```

#### 4. Test Bindings

```bash
cd ../auroraai-router/cactus-final

# Test if Cactus library loads correctly
uv run python bindings/test_bindings.py

# Test with actual model
uv run python bindings/test_bindings.py \
    --model ../../cactus/weights/LFM2-350M/model.gguf
```

#### 5. Generate Production Profile

```bash
# Full training with real Cactus embeddings
uv run python training/generate_profile.py \
    --use-cactus \
    --model-path ../../cactus/weights/LFM2-350M/model.gguf \
    --output profiles/production_profile.json

# With custom library path
uv run python training/generate_profile.py \
    --use-cactus \
    --model-path ../../cactus/weights/LFM2-350M/model.gguf \
    --lib-path ../../cactus/build/cactus/libcactus.dylib \
    --output profiles/production_profile.json
```

**⏱️ Expected time:** 10-30 minutes depending on your Mac

---

## 🔧 Configuration

### 12 Cactus Models

The router is trained to route between these 12 models:

| Model ID | Size | Speed | Capabilities | Embed Support |
|----------|------|-------|--------------|---------------|
| gemma-270m | 172MB | 173 tok/s | text | ❌ |
| lfm2-350m | 233MB | 145 tok/s | text, tools, embed | ✅ |
| smollm-360m | 227MB | 150 tok/s | text | ❌ |
| qwen-600m | 394MB | 129 tok/s | text, tools, embed | ✅ |
| lfm2-vl-450m | 420MB | 113 tok/s | text, vision, embed | ✅ |
| lfm2-700m | 467MB | 115 tok/s | text, tools, embed | ✅ |
| gemma-1b | 642MB | 100 tok/s | text | ❌ |
| lfm2-1.2b | 722MB | 95 tok/s | text, tools, embed | ✅ |
| lfm2-1.2b-tools | 722MB | 95 tok/s | text, tools, embed | ✅ |
| qwen-1.7b | 1161MB | 75 tok/s | text, tools, embed | ✅ |
| smollm-1.7b | 1161MB | 72 tok/s | text, embed | ✅ |
| lfm2-vl-1.6b | 1440MB | 60 tok/s | text, vision, embed | ✅ |

*Benchmarks from Mac M4 Pro*

### MMLU Topics

The router is trained on 15 diverse MMLU topics:

- **Math:** abstract_algebra
- **Medical:** anatomy, professional_medicine
- **Science:** astronomy, high_school_physics, virology
- **Engineering:** electrical_engineering
- **Social Sciences:** world_religions, philosophy, econometrics
- **Humanities:** international_law, marketing, high_school_geography
- **CS:** computer_security
- **Ethics:** moral_scenarios

~150 samples per topic = ~2250 total samples

### Clustering Algorithms

- **KMeans:** Tests K from 5 to 15
- **HDBSCAN:** Tests various `min_cluster_size` and `min_samples` values
- **Selection:** Best algorithm chosen by silhouette score (cosine distance)

---

## 📦 Profile Format

Generated profiles are JSON files (~100KB) with this structure:

```json
{
  "version": "2.0",
  "metadata": {
    "n_clusters": 10,
    "feature_dim": 768,
    "embedding_model": "lfm2-350m",
    "silhouette_score": 0.4767,
    "clustering_algorithm": "HDBSCAN",
    "target": "cactus_compute",
    "dataset": "mmlu",
    "n_samples": 2065,
    "is_mock": false
  },
  "cluster_centers": {
    "n_clusters": 10,
    "feature_dim": 768,
    "cluster_centers": [[...], [...], ...],
    "dtype": "float16"
  },
  "llm_profiles": {
    "gemma-270m": [0.53, 0.51, ...],
    "lfm2-350m": [0.52, 0.49, ...],
    ...
  },
  "models": [...]
}
```

**Key fields:**
- `cluster_centers`: Float16 centroids for fast cosine similarity
- `llm_profiles`: Error rate per model per cluster
- `models`: Full model metadata (size, speed, capabilities)
- `is_mock`: Flag indicating if embeddings are mock (for validation)

---

## 🧪 Testing

### 1. Test Cactus Bindings

```bash
# Check if Cactus is available
uv run python bindings/test_bindings.py

# Test with model (Mac only)
uv run python bindings/test_bindings.py --model path/to/model.gguf
```

### 2. View Model Configuration

```bash
# Print all 12 models
uv run python training/config.py
```

### 3. Test Profile Generation (Mock)

```bash
# Quick test on x86 with mock embeddings
# First install mock dependencies
uv sync --extra mock

# Then run
uv run python training/generate_profile.py \
    --mock-embeddings \
    --output profiles/test_profile.json
```

---

## 🔬 Next Steps (After Profile Generation)

Once you have a production profile, you can:

1. **Integrate with Router SDK:**
   ```python
   from auroraai_router import AuroraAIRouter

   router = AuroraAIRouter('profiles/production_profile.json')
   result = router.route("Explain quantum entanglement", cost_preference=0.7)

   # Use result.model_path with Cactus
   import cactus
   model = cactus.init(result.model_path, 2048)
   response = cactus.complete(model, messages)
   ```

2. **Replace Error Rate Simulation:**

   Currently, error rates are simulated based on model size. For production, replace with real Cactus inference:

   ```python
   # In utils.py, replace simulate_model_error_rate() with:
   def compute_real_error_rate(model_path, cluster_samples):
       model = cactus.init(model_path, 2048)
       correct = 0
       for sample in cluster_samples:
           response = cactus.complete(model, sample['question'])
           if is_correct_answer(response, sample['answer']):
               correct += 1
       return 1.0 - (correct / len(cluster_samples))
   ```

3. **Deploy to Mobile:**
   - Profile is ~100KB, perfect for mobile
   - Load profile in your app
   - Use Cactus SDK for routing + inference

---

## 🛠️ Dependencies

Managed via `pyproject.toml` with uv:

**Core dependencies:**
- `datasets` - MMLU dataset loading
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `scikit-learn` - KMeans clustering
- `hdbscan` - HDBSCAN clustering
- `tqdm` - Progress bars

**Optional extras:**
- `[mock]` - sentence-transformers for x86 testing
- `[viz]` - matplotlib, plotly, umap for visualizations
- `[dev]` - pytest, black, ruff for development

**Install:**
```bash
# Core dependencies only
uv sync

# With mock embeddings support
uv sync --extra mock

# With all extras
uv sync --all-extras
```

---

## 🐛 Troubleshooting

### "Cactus library not found"

**Problem:** Can't find `libcactus.dylib`

**Solution:**
```bash
# Build Cactus first
cd cactus
./apple/build.sh

# Or specify path explicitly
python training/generate_profile.py \
    --use-cactus \
    --lib-path /path/to/libcactus.dylib \
    --model-path models/lfm2-350m.gguf
```

### "ARM architecture required"

**Problem:** Trying to use `--use-cactus` on x86

**Solution:** Use `--mock-embeddings` for testing on x86, or run on your Mac

### "Model file not found"

**Problem:** Model path incorrect

**Solution:**
```bash
# Download model using Cactus CLI
cd cactus
./cli/cactus download LiquidAI/LFM2-350M

# Use correct path
python training/generate_profile.py \
    --use-cactus \
    --model-path ../cactus/weights/LFM2-350M/model.gguf
```

### Out of memory during clustering

**Problem:** Too many samples for RAM

**Solution:** Reduce samples in config.py:
```python
MMLU_CONFIG = {
    'samples_per_topic': 100,  # Reduce from 150
}
```

---

## 📚 References

- **Cactus Repository:** https://github.com/cactus-compute/cactus
- **MMLU Dataset:** https://huggingface.co/datasets/cais/mmlu
- **COLAB Notebook:** `notebooks/COLAB_profiling.ipynb`
- **Previous Integration:** `cactus-integration/` (AWS-based)

---

## 📝 TODO

- [ ] Replace simulated error rates with real Cactus inference
- [ ] Add visualization generation (cluster plots, heatmaps)
- [ ] Add profile validation script
- [ ] Add benchmark script to compare router performance
- [ ] Create router SDK integration example
- [ ] Add support for custom datasets beyond MMLU

---

## 📄 License

MIT License - Same as AuroraAI Router

---

## 🎉 Summary

This is a **clean restart** focused on:
✅ **Profile generation only** (no routing logic yet)
✅ **Cactus C bindings** via ctypes
✅ **Mac-ready** training pipeline
✅ **Mock mode** for x86 testing
✅ **Production profiles** (~100KB JSON)

Next step: **Run on your Mac** to generate real production profiles! 🚀
