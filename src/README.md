# AuroraAI Router - Source Code

Intelligent query router for local LLM selection using embedding-based clustering.

## 📁 Directory Structure

```
src/
├── __init__.py          # Package root with unified API
├── __main__.py          # CLI entry point
├── config.py            # Shared configuration (models, paths, embeddings)
├── router_factory.py    # Unified router factory
│
├── v1/                  # Original Router (MiniLM + KMeans)
│   ├── __init__.py
│   ├── embedder.py      # MiniLMEmbedder (384-dim, ~80MB)
│   └── router.py        # RouterV1 implementation
│
├── v2/                  # CACTUS Router (Nomic + HDBSCAN)
│   ├── __init__.py
│   ├── embedder.py      # NomicEmbedder (768-dim, Matryoshka)
│   ├── cluster_engine.py # HDBSCAN clustering engine
│   └── router.py        # CactusRouter implementation
│
├── training/            # Profile training scripts
│   ├── __init__.py
│   ├── train_v1.py      # V1 KMeans trainer
│   └── train_v2.py      # V2 CACTUS trainer
│
└── benchmarks/          # Benchmarking and evaluation
    ├── __init__.py
    ├── baselines.py     # Baseline routers for comparison
    ├── benchmark.py     # RouterBenchmark class
    └── compare.py       # V1 vs V2 comparison
```

## 🚀 Quick Start

### Using the Unified API

```python
from src import create_router

# Auto-detects profile version and creates appropriate router
router = create_router(profile_path="profiles/cactus_profile.json")

# Route a query
result = router.route("What is quantum entanglement?")
print(f"Selected model: {result['model']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Using V1 Router Directly

```python
from src.v1 import RouterV1, MiniLMEmbedder

# Create embedder
embedder = MiniLMEmbedder()

# Create router
router = RouterV1(profile_path="profiles/v1_profile.json", embedder=embedder)

result = router.route("Explain gravity")
```

### Using V2 CACTUS Router Directly

```python
from src.v2 import CactusRouter, NomicEmbedder

# Create embedder
embedder = NomicEmbedder()

# Create router
router = CactusRouter(profile_path="profiles/cactus_profile.json", embedder=embedder)

result = router.route("What is machine learning?")
```

## 📊 Training

### Train V2 (CACTUS) Profile

```python
from src.training import CactusTrainer

trainer = CactusTrainer()
profile_path = trainer.train(
    output_path="profiles/my_profile.json",
    n_samples=2000,
    use_gpu=True
)
```

### Train V1 Profile

```python
from src.training import TrainerV1

trainer = TrainerV1()
profile_path = trainer.train(
    output_path="profiles/v1_profile.json",
    n_samples=2000
)
```

## 📈 Benchmarking

### Benchmark a Router

```python
from src.benchmarks import RouterBenchmark
from src import create_router

router = create_router("profiles/cactus_profile.json")
benchmark = RouterBenchmark()

results = benchmark.run_benchmark(router)
print(results.summary())
```

### Compare V1 vs V2

```python
from src.benchmarks import compare_routers

result = compare_routers(
    v1_profile="profiles/v1_profile.json",
    v2_profile="profiles/cactus_profile.json",
    output_dir="comparison_results"
)

print(f"Accuracy improvement: {result.accuracy_delta:+.2%}")
```

## 🖥️ Command Line Interface

```bash
# Train a new V2 profile
python -m src train --version v2 --output profiles/my_profile.json --samples 2000

# Benchmark a profile
python -m src benchmark --profile profiles/cactus_profile.json

# Route a query
python -m src route --profile profiles/cactus_profile.json "What is AI?"

# Compare routers
python -m src compare --v2 profiles/cactus_profile.json --output results/

# Show profile information
python -m src info --profile profiles/cactus_profile.json
```

## ⚙️ Configuration

### Available Models (CACTUS_MODELS)

| Model | Size (MB) | Use Case |
|-------|-----------|----------|
| gemma-270m | 172 | Simple queries |
| qwen-0.6b | 411 | Basic reasoning |
| gemma-450m | 293 | Moderate complexity |
| qwen-1.8b | 1240 | Complex reasoning |
| lfm2-vl-1.6b | 1440 | Vision + Language |
| ... | ... | ... |

### Embedding Models

| Version | Model | Dimensions | Size |
|---------|-------|------------|------|
| V1 | all-MiniLM-L6-v2 | 384 | ~80MB |
| V2 | nomic-embed-text-v1.5 | 768 | ~550MB |

### Clustering Algorithms

| Version | Algorithm | Parameters |
|---------|-----------|------------|
| V1 | KMeans | n_clusters=8 |
| V2 | HDBSCAN | min_cluster_size=20, min_samples=15 |

## 📊 Profile Format

### V2 (CACTUS) Profile Structure

```json
{
  "version": "2.0",
  "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
  "embedding_dim": 768,
  "cluster_centers": [[...], [...], ...],
  "cluster_assignments": [0, 1, 2, 0, ...],
  "model_error_rates": {
    "0": {"gemma-270m": 0.15, ...},
    "1": {"gemma-270m": 0.08, ...}
  },
  "models": [...],
  "metadata": {
    "n_clusters": 5,
    "silhouette_score": 0.42,
    "created_at": "2025-01-01T00:00:00"
  }
}
```

## 🔧 Dependencies

### Core
- `numpy` - Numerical operations
- `torch` - Embedding model inference
- `transformers` - Hugging Face models

### V1 Specific
- `sentence-transformers` - MiniLM embeddings
- `scikit-learn` - KMeans clustering

### V2 Specific
- `hdbscan` - Density-based clustering
- `umap-learn` - Dimensionality reduction (optional)

### Training
- `datasets` - MMLU dataset loading
- `tqdm` - Progress bars

### Benchmarking
- `matplotlib` - Visualizations
- `scipy` - Statistical tests

## 📝 Migration Guide (V1 → V2)

1. **Update profile path**: V2 profiles have different structure
2. **Change imports**:
   ```python
   # Old
   from router import MobileRouter
   
   # New
   from src import create_router
   router = create_router(profile_path)
   ```
3. **Handle embedding dimension change**: 384 → 768

## 🤝 Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](../LICENSE)
