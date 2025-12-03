"""
Configuration for Cactus models and training parameters.

This module defines all 12 Cactus models that the router will support,
along with their capabilities and expected performance characteristics.
"""

# 12 Cactus models for local routing
# Source: https://github.com/cactus-compute/cactus README.md
CACTUS_MODELS = [
    {
        'model_id': 'gemma-270m',
        'model_path': 'google/gemma-3-270m-it',
        'hf_name': 'google/gemma-3-270m-it',
        'size_mb': 172,
        'avg_tokens_per_sec': 173,  # M4 Pro benchmark
        'capabilities': ['text'],
        'context_size': 2048,
        'has_embed': False,
        'description': 'Smallest Gemma model, good for simple tasks'
    },
    {
        'model_id': 'lfm2-350m',
        'model_path': 'LiquidAI/LFM2-350M',
        'hf_name': 'LiquidAI/LFM2-350M',
        'size_mb': 233,
        'avg_tokens_per_sec': 145,
        'capabilities': ['text', 'tools', 'embed'],
        'context_size': 2048,
        'has_embed': True,
        'description': 'Small LFM2 with tool calling and embedding support'
    },
    {
        'model_id': 'smollm-360m',
        'model_path': 'HuggingFaceTB/SmolLM2-360m-Instruct',
        'hf_name': 'HuggingFaceTB/SmolLM2-360m-Instruct',
        'size_mb': 227,
        'avg_tokens_per_sec': 150,
        'capabilities': ['text'],
        'context_size': 2048,
        'has_embed': False,
        'description': 'Efficient small language model'
    },
    {
        'model_id': 'qwen-600m',
        'model_path': 'Qwen/Qwen3-0.6B',
        'hf_name': 'Qwen/Qwen3-0.6B',
        'size_mb': 394,
        'avg_tokens_per_sec': 129,
        'capabilities': ['text', 'tools', 'embed'],
        'context_size': 2048,
        'has_embed': True,
        'description': 'Qwen 600M with tool calling and embedding'
    },
    {
        'model_id': 'lfm2-vl-450m',
        'model_path': 'LiquidAI/LFM2-VL-450M',
        'hf_name': 'LiquidAI/LFM2-VL-450M',
        'size_mb': 420,
        'avg_tokens_per_sec': 113,
        'capabilities': ['text', 'vision', 'embed'],
        'context_size': 2048,
        'has_embed': True,
        'description': 'Vision-language model for image understanding'
    },
    {
        'model_id': 'lfm2-700m',
        'model_path': 'LiquidAI/LFM2-700M',
        'hf_name': 'LiquidAI/LFM2-700M',
        'size_mb': 467,
        'avg_tokens_per_sec': 115,
        'capabilities': ['text', 'tools', 'embed'],
        'context_size': 2048,
        'has_embed': True,
        'description': 'Larger LFM2 for better performance'
    },
    {
        'model_id': 'gemma-1b',
        'model_path': 'google/gemma-3-1b-it',
        'hf_name': 'google/gemma-3-1b-it',
        'size_mb': 642,
        'avg_tokens_per_sec': 100,
        'capabilities': ['text'],
        'context_size': 2048,
        'has_embed': False,
        'description': '1B Gemma model for balanced performance'
    },
    {
        'model_id': 'lfm2-1.2b',
        'model_path': 'LiquidAI/LFM2-1.2B',
        'hf_name': 'LiquidAI/LFM2-1.2B',
        'size_mb': 722,
        'avg_tokens_per_sec': 95,
        'capabilities': ['text', 'tools', 'embed'],
        'context_size': 2048,
        'has_embed': True,
        'description': 'High-performance 1.2B LFM2 model'
    },
    {
        'model_id': 'lfm2-1.2b-tools',
        'model_path': 'LiquidAI/LFM2-1.2B-Tools',
        'hf_name': 'LiquidAI/LFM2-1.2B-Tools',
        'size_mb': 722,
        'avg_tokens_per_sec': 95,
        'capabilities': ['text', 'tools', 'embed'],
        'context_size': 2048,
        'has_embed': True,
        'description': 'LFM2 1.2B specialized for tool calling'
    },
    {
        'model_id': 'qwen-1.7b',
        'model_path': 'Qwen/Qwen3-1.7B',
        'hf_name': 'Qwen/Qwen3-1.7B',
        'size_mb': 1161,
        'avg_tokens_per_sec': 75,
        'capabilities': ['text', 'tools', 'embed'],
        'context_size': 2048,
        'has_embed': True,
        'description': 'Powerful 1.7B Qwen model'
    },
    {
        'model_id': 'smollm-1.7b',
        'model_path': 'HuggingFaceTB/SmolLM2-1.7B-Instruct',
        'hf_name': 'HuggingFaceTB/SmolLM2-1.7B-Instruct',
        'size_mb': 1161,
        'avg_tokens_per_sec': 72,
        'capabilities': ['text', 'embed'],
        'context_size': 2048,
        'has_embed': True,
        'description': 'Larger SmolLM for better quality'
    },
    {
        'model_id': 'lfm2-vl-1.6b',
        'model_path': 'LiquidAI/LFM2-VL-1.6B',
        'hf_name': 'LiquidAI/LFM2-VL-1.6B',
        'size_mb': 1440,
        'avg_tokens_per_sec': 60,
        'capabilities': ['text', 'vision', 'embed'],
        'context_size': 2048,
        'has_embed': True,
        'description': 'Largest vision-language model'
    },
]

# Models that can be used for embedding extraction
EMBEDDING_MODELS = [m for m in CACTUS_MODELS if m['has_embed']]

# Recommended embedding model (good balance of size and quality)
DEFAULT_EMBEDDING_MODEL = 'lfm2-350m'

# MMLU dataset configuration
MMLU_CONFIG = {
    'dataset_name': 'cais/mmlu',
    'subset': 'all',
    'split': 'test',
    'topics': [
        'abstract_algebra',        # Math
        'anatomy',                 # Medical
        'world_religions',         # Religion
        'computer_security',       # CS
        'astronomy',               # Space/Physics
        'international_law',       # Law
        'marketing',               # Business
        'high_school_geography',   # Geography
        'philosophy',              # Philosophy
        'electrical_engineering',  # Engineering
        'high_school_physics',     # Physics
        'econometrics',            # Economics
        'moral_scenarios',         # Ethics
        'professional_medicine',   # Medicine
        'virology',                # Biology
    ],
    'samples_per_topic': 150,  # ~2250 samples total
}

# Clustering configuration
CLUSTERING_CONFIG = {
    'kmeans_k_range': range(5, 16),  # Test K from 5 to 15
    'hdbscan_params': [
        {'min_cluster_size': 20, 'min_samples': 5},
        {'min_cluster_size': 20, 'min_samples': 10},
        {'min_cluster_size': 20, 'min_samples': 15},
        {'min_cluster_size': 30, 'min_samples': 5},
        {'min_cluster_size': 30, 'min_samples': 10},
        {'min_cluster_size': 30, 'min_samples': 15},
        {'min_cluster_size': 50, 'min_samples': 5},
        {'min_cluster_size': 50, 'min_samples': 10},
        {'min_cluster_size': 50, 'min_samples': 15},
    ],
    'metric': 'cosine',  # Use cosine distance for embeddings
    'random_state': 42,
}

# Training configuration
TRAINING_CONFIG = {
    'random_seed': 42,
    'embedding_dtype': 'float16',  # Save space while maintaining precision
    'cache_embeddings': True,  # Cache embeddings to disk for faster re-runs
    'batch_size': 32,  # Batch size for embedding generation
    'verbose': True,
}

# Profile output configuration
PROFILE_CONFIG = {
    'version': '2.0',
    'format': 'json',
    'indent': 2,
    'save_visualizations': True,  # Save cluster plots and heatmaps
    'save_statistics': True,  # Include detailed training stats
}


def get_model_by_id(model_id: str) -> dict:
    """Get model configuration by model_id."""
    for model in CACTUS_MODELS:
        if model['model_id'] == model_id:
            return model
    raise ValueError(f"Model not found: {model_id}")


def get_embedding_model_ids() -> list[str]:
    """Get list of model IDs that support embeddings."""
    return [m['model_id'] for m in EMBEDDING_MODELS]


def print_models():
    """Print all available models in a formatted table."""
    print("\n" + "=" * 100)
    print(f"{'Model ID':<20} {'Size':>8} {'Speed':>10} {'Capabilities':<30} Description")
    print("=" * 100)

    for model in CACTUS_MODELS:
        model_id = model['model_id']
        size = f"{model['size_mb']}MB"
        speed = f"{model['avg_tokens_per_sec']} tok/s"
        caps = ', '.join(model['capabilities'])
        desc = model['description'][:40]

        print(f"{model_id:<20} {size:>8} {speed:>10} {caps:<30} {desc}")

    print("=" * 100)
    print(f"\nTotal models: {len(CACTUS_MODELS)}")
    print(f"Models with embedding support: {len(EMBEDDING_MODELS)}")


if __name__ == "__main__":
    # Print model table
    print_models()

    # Print embedding models
    print("\nModels with Embedding Support:")
    for model_id in get_embedding_model_ids():
        model = get_model_by_id(model_id)
        print(f"  - {model_id:<20} ({model['size_mb']}MB)")

    print(f"\nDefault embedding model: {DEFAULT_EMBEDDING_MODEL}")
