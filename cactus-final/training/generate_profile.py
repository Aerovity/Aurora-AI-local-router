"""
AuroraAI Router Profile Generation with Cactus Embeddings

This script generates router profiles using MMLU dataset and Cactus embeddings.
It follows the strategy from the COLAB notebook:
1. Load MMLU dataset (~2000 samples, 15 topics)
2. Extract embeddings using Cactus C library
3. Cluster embeddings (KMeans + HDBSCAN)
4. Compute error rates per cluster per model
5. Save production profile (~100KB JSON)

Usage:
    # On x86 (mock mode for testing structure)
    python training/generate_profile.py --mock-embeddings --output profiles/test_profile.json

    # On Mac (real Cactus embeddings)
    python training/generate_profile.py \
        --use-cactus \
        --model-path models/lfm2-350m-q8.gguf \
        --output profiles/production_profile.json
"""

import argparse
import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import (
    CACTUS_MODELS,
    MMLU_CONFIG,
    CLUSTERING_CONFIG,
    TRAINING_CONFIG,
    PROFILE_CONFIG,
    get_model_by_id,
)
from training.utils import (
    compute_cluster_error_rates,
    save_profile,
    print_cluster_distribution,
    save_embeddings_cache,
    load_embeddings_cache,
    Timer,
)


def set_random_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    print(f"🎲 Random seed set to: {seed}")


def load_mmlu_dataset(
    topics: List[str],
    samples_per_topic: int = 150
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Load MMLU dataset samples.

    Args:
        topics: List of MMLU topics to sample from
        samples_per_topic: Number of samples per topic

    Returns:
        Tuple of (samples_list, dataframe)
    """
    print("\n" + "=" * 80)
    print("📚 Loading MMLU Dataset")
    print("=" * 80)

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library not found. Install with: pip install datasets"
        )

    with Timer("Loading MMLU"):
        mmlu = load_dataset(MMLU_CONFIG['dataset_name'], MMLU_CONFIG['subset'])

    # Sample from each topic
    samples = []
    for topic in topics:
        topic_samples = [x for x in mmlu[MMLU_CONFIG['split']] if x['subject'] == topic]
        n_to_sample = min(samples_per_topic, len(topic_samples))
        samples.extend(random.sample(topic_samples, n_to_sample))

    random.shuffle(samples)

    print(f"\n✅ Loaded {len(samples)} samples from {len(topics)} topics")
    print(f"\n📋 Topic Distribution:")
    for topic in topics:
        count = sum(1 for s in samples if s['subject'] == topic)
        print(f"  {topic:30s}: {count:3d} samples")

    # Create dataframe
    df = pd.DataFrame({
        'question': [s['question'] for s in samples],
        'subject': [s['subject'] for s in samples],
        'choices': [s['choices'] for s in samples],
        'answer': [s['answer'] for s in samples],
    })

    return samples, df


def extract_embeddings_mock(texts: List[str], embedding_dim: int = 768) -> np.ndarray:
    """
    Generate mock embeddings using sentence-transformers (for x86 testing).

    Args:
        texts: List of text strings
        embedding_dim: Target embedding dimension

    Returns:
        Embeddings array of shape (len(texts), embedding_dim)
    """
    print("\n⚠️  Using MOCK embeddings (sentence-transformers)")
    print("   These are NOT Cactus embeddings!")
    print("   Run on Mac with --use-cactus for real embeddings.\n")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not found. Install with: pip install sentence-transformers"
        )

    # Use a model that produces 768-dim embeddings (compatible with Cactus)
    model_name = "BAAI/bge-base-en-v1.5"

    print(f"📥 Loading embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)

    print(f"🔢 Extracting embeddings for {len(texts)} texts...")
    embeddings = embedder.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True
    )

    print(f"✅ Embeddings extracted: {embeddings.shape}")
    return embeddings


def extract_embeddings_cactus(
    texts: List[str],
    model_path: str,
    lib_path: str = None
) -> np.ndarray:
    """
    Generate embeddings using Cactus C library.

    Args:
        texts: List of text strings
        model_path: Path to Cactus GGUF model
        lib_path: Optional path to libcactus library

    Returns:
        Embeddings array of shape (len(texts), embedding_dim)
    """
    from bindings import CactusModel, CactusNotAvailableError

    print("\n🌵 Using REAL Cactus embeddings")
    print(f"   Model: {model_path}\n")

    try:
        with Timer("Extracting Cactus embeddings"):
            model = CactusModel(model_path, context_size=2048, lib_path=lib_path)
            embeddings = model.embed_batch(texts, show_progress=True)
            model.destroy()

        print(f"\n✅ Embeddings extracted: {embeddings.shape}")
        return embeddings

    except CactusNotAvailableError as e:
        print(f"\n❌ {e}")
        print("\nPlease run this script with --mock-embeddings on x86,")
        print("or run on your Mac with --use-cactus.")
        sys.exit(1)


def cluster_embeddings(
    embeddings: np.ndarray,
    kmeans_k_range: range,
    hdbscan_params: List[Dict],
    metric: str = 'cosine'
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Cluster embeddings using KMeans and HDBSCAN, return best clustering.

    Args:
        embeddings: Embeddings array
        kmeans_k_range: Range of K values to test for KMeans
        hdbscan_params: List of HDBSCAN parameter configurations
        metric: Distance metric ('cosine' or 'euclidean')

    Returns:
        Tuple of (labels, centroids, best_config)
    """
    print("\n" + "=" * 80)
    print("🔍 Testing Clustering Algorithms")
    print("=" * 80)

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        import hdbscan
    except ImportError:
        raise ImportError(
            "Clustering libraries not found. Install with: "
            "pip install scikit-learn hdbscan"
        )

    results = []

    # Test KMeans
    print("\n📊 Testing KMeans:")
    for k in kmeans_k_range:
        with Timer(f"KMeans K={k}", verbose=False):
            kmeans = KMeans(
                n_clusters=k,
                random_state=CLUSTERING_CONFIG['random_state'],
                n_init=10
            )
            labels = kmeans.fit_predict(embeddings)
            sil = silhouette_score(embeddings, labels, metric=metric)

        results.append({
            'algo': 'KMeans',
            'k': k,
            'silhouette': sil,
            'labels': labels,
            'centroids': kmeans.cluster_centers_
        })
        print(f"  K={k:2d}: silhouette={sil:.4f}")

    # Test HDBSCAN
    print("\n📊 Testing HDBSCAN:")
    for params in hdbscan_params:
        mcs = params['min_cluster_size']
        ms = params['min_samples']

        with Timer(f"HDBSCAN mcs={mcs}, ms={ms}", verbose=False):
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                metric='euclidean'  # HDBSCAN uses euclidean
            )
            labels = clusterer.fit_predict(embeddings)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()

        # Compute silhouette (exclude noise)
        if n_clusters >= 2 and (labels != -1).sum() > n_clusters:
            mask = labels != -1
            sil = silhouette_score(embeddings[mask], labels[mask], metric=metric)
        else:
            sil = -1

        # Compute centroids (exclude noise)
        if n_clusters > 0:
            unique_labels = sorted(set(labels) - {-1})
            centroids = np.array([embeddings[labels == i].mean(axis=0) for i in unique_labels])
        else:
            centroids = None

        results.append({
            'algo': 'HDBSCAN',
            'k': n_clusters,
            'silhouette': sil,
            'labels': labels,
            'centroids': centroids,
            'params': params,
            'noise': n_noise
        })

        print(f"  mcs={mcs}, ms={ms:2d}: K={n_clusters:2d}, noise={n_noise:4d}, sil={sil:.4f}")

    # Find best configuration
    best = max(results, key=lambda x: x['silhouette'])

    print("\n" + "=" * 80)
    print(f"🏆 BEST: {best['algo']} with K={best['k']}, Silhouette={best['silhouette']:.4f}")
    print("=" * 80)

    return best['labels'], best['centroids'], best


def main():
    parser = argparse.ArgumentParser(
        description="Generate AuroraAI Router profile with Cactus embeddings"
    )

    # Embedding mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--use-cactus',
        action='store_true',
        help='Use real Cactus embeddings (requires Mac/ARM)'
    )
    mode_group.add_argument(
        '--mock-embeddings',
        action='store_true',
        help='Use mock embeddings for testing on x86'
    )

    # Model configuration (for Cactus mode)
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to Cactus GGUF model (required with --use-cactus)'
    )
    parser.add_argument(
        '--lib-path',
        type=str,
        help='Path to libcactus shared library (optional)'
    )

    # Output
    parser.add_argument(
        '--output',
        type=str,
        default='profiles/router_profile.json',
        help='Output profile path (default: profiles/router_profile.json)'
    )

    # Dataset configuration
    parser.add_argument(
        '--n-samples',
        type=int,
        default=None,
        help='Total number of samples (default: all topics, ~2000 samples)'
    )

    # Caching
    parser.add_argument(
        '--cache-embeddings',
        action='store_true',
        default=True,
        help='Cache embeddings to disk (default: True)'
    )
    parser.add_argument(
        '--use-cache',
        action='store_true',
        help='Use cached embeddings if available'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.use_cactus and not args.model_path:
        parser.error("--model-path is required when using --use-cactus")

    # Set random seed
    set_random_seed(TRAINING_CONFIG['random_seed'])

    # Load MMLU dataset
    samples, df = load_mmlu_dataset(
        topics=MMLU_CONFIG['topics'],
        samples_per_topic=MMLU_CONFIG['samples_per_topic']
    )

    texts = [s['question'] for s in samples]

    # Extract embeddings
    print("\n" + "=" * 80)
    print("🔢 Extracting Embeddings")
    print("=" * 80)

    cache_file = Path('data/embeddings_cache.npz')
    embeddings = None

    # Try loading from cache
    if args.use_cache and cache_file.exists():
        embeddings, metadata = load_embeddings_cache(cache_file)

    # Generate embeddings
    if embeddings is None:
        if args.use_cactus:
            embeddings = extract_embeddings_cactus(
                texts,
                model_path=args.model_path,
                lib_path=args.lib_path
            )
            embedding_model = Path(args.model_path).stem
        else:
            embeddings = extract_embeddings_mock(texts)
            embedding_model = "bge-base-en-v1.5 (MOCK)"

        # Cache embeddings
        if args.cache_embeddings:
            save_embeddings_cache(
                embeddings,
                cache_file,
                metadata={'model': embedding_model, 'n_samples': len(texts)}
            )

    # Convert to float16 to save space
    embeddings = embeddings.astype(np.float16)

    # Cluster embeddings
    labels, centroids, best_config = cluster_embeddings(
        embeddings.astype(np.float32),  # Use float32 for clustering
        kmeans_k_range=CLUSTERING_CONFIG['kmeans_k_range'],
        hdbscan_params=CLUSTERING_CONFIG['hdbscan_params'],
        metric=CLUSTERING_CONFIG['metric']
    )

    # Print cluster distribution
    print_cluster_distribution(labels, topics=MMLU_CONFIG['topics'])

    # Compute error rates
    error_rates = compute_cluster_error_rates(
        models=CACTUS_MODELS,
        cluster_labels=labels,
        samples=samples
    )

    # Create profile
    print("\n" + "=" * 80)
    print("💾 Creating Router Profile")
    print("=" * 80)

    unique_clusters = sorted(set(labels) - {-1})

    profile = {
        'version': PROFILE_CONFIG['version'],
        'metadata': {
            'n_clusters': len(unique_clusters),
            'feature_dim': embeddings.shape[1],
            'embedding_model': embedding_model if args.use_cactus else "bge-base-en-v1.5 (MOCK)",
            'lambda_min': 0.0,
            'lambda_max': 2.0,
            'default_cost_preference': 0.5,
            'silhouette_score': float(best_config['silhouette']),
            'clustering_algorithm': best_config['algo'],
            'target': 'cactus_compute',
            'dataset': 'mmlu',
            'n_samples': len(samples),
            'topics': MMLU_CONFIG['topics'],
            'is_mock': not args.use_cactus,  # Flag to indicate if embeddings are mock
        },
        'cluster_centers': {
            'n_clusters': len(unique_clusters),
            'feature_dim': centroids.shape[1],
            'cluster_centers': centroids.astype(np.float16).tolist(),
            'dtype': 'float16'
        },
        'llm_profiles': error_rates,
        'models': CACTUS_MODELS
    }

    # Save profile
    save_profile(profile, args.output, indent=PROFILE_CONFIG['indent'])

    print("\n" + "=" * 80)
    print("✅ Profile Generation Complete!")
    print("=" * 80)
    print(f"\n📊 Profile Summary:")
    print(f"  Models: {len(CACTUS_MODELS)}")
    print(f"  Clusters: {len(unique_clusters)}")
    print(f"  Samples: {len(samples)}")
    print(f"  Embedding Model: {embedding_model if args.use_cactus else 'bge-base-en-v1.5 (MOCK)'}")
    print(f"  Silhouette Score: {best_config['silhouette']:.4f}")
    print(f"  Algorithm: {best_config['algo']}")

    if not args.use_cactus:
        print("\n⚠️  WARNING: This profile uses MOCK embeddings!")
        print("   For production use, regenerate on Mac with --use-cactus")

    print(f"\n📁 Profile saved to: {args.output}")


if __name__ == "__main__":
    main()
