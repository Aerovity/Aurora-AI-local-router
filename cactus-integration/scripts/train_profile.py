"""
Train AuroraAI Router Profile with Cactus Embeddings

This script trains a routing profile using real Cactus embeddings,
ensuring cluster centers match exactly what mobile devices produce.

Usage:
    python train_profile.py

Output:
    profiles/cactus_router_profile.json
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cactus_wrapper import CactusEmbedder

# =============================================================================
# Configuration
# =============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths (relative to ~/cactus-integration/)
HOME = os.path.expanduser("~")
BASE_DIR = f"{HOME}/cactus-integration"
MODEL_PATH = f"{BASE_DIR}/cactus/weights/lfm2-350m"
LIB_PATH = f"{BASE_DIR}/lib/libcactus.so"
OUTPUT_DIR = f"{BASE_DIR}/profiles"

# Training config
N_SAMPLES = 2000  # Number of MMLU samples to use
SAMPLES_PER_TOPIC = 150  # Samples per topic

# MMLU topics (diverse domains)
TOPICS = [
    "abstract_algebra",        # Math
    "anatomy",                 # Medical
    "world_religions",         # Religion
    "computer_security",       # CS
    "astronomy",               # Space/Physics
    "international_law",       # Law
    "marketing",               # Business
    "high_school_geography",   # Geography
    "philosophy",              # Philosophy
    "electrical_engineering",  # Engineering
    "high_school_physics",     # Physics
    "econometrics",            # Economics
    "moral_scenarios",         # Ethics
    "professional_medicine",   # Medicine
    "virology",                # Biology
]

# Cactus models to route between
CACTUS_MODELS = [
    {'model_id': 'gemma-270m', 'size_mb': 172, 'tokens_per_sec': 173},
    {'model_id': 'lfm2-350m', 'size_mb': 233, 'tokens_per_sec': 145},
    {'model_id': 'smollm-360m', 'size_mb': 227, 'tokens_per_sec': 150},
    {'model_id': 'qwen-600m', 'size_mb': 394, 'tokens_per_sec': 129},
    {'model_id': 'lfm2-700m', 'size_mb': 467, 'tokens_per_sec': 115},
    {'model_id': 'gemma-1b', 'size_mb': 642, 'tokens_per_sec': 100},
    {'model_id': 'lfm2-1.2b', 'size_mb': 722, 'tokens_per_sec': 95},
    {'model_id': 'qwen-1.7b', 'size_mb': 1161, 'tokens_per_sec': 75},
    {'model_id': 'smollm-1.7b', 'size_mb': 1161, 'tokens_per_sec': 72},
    {'model_id': 'lfm2-vl-1.6b', 'size_mb': 1440, 'tokens_per_sec': 60},
]


# =============================================================================
# Data Loading
# =============================================================================

def load_mmlu_samples() -> List[Dict]:
    """Load diverse samples from MMLU dataset."""
    print("📚 Loading MMLU dataset...")
    
    from datasets import load_dataset
    mmlu = load_dataset("cais/mmlu", "all")
    
    samples = []
    for topic in TOPICS:
        topic_samples = [x for x in mmlu["test"] if x["subject"] == topic]
        n_to_sample = min(SAMPLES_PER_TOPIC, len(topic_samples))
        samples.extend(random.sample(topic_samples, n_to_sample))
        print(f"   {topic}: {n_to_sample} samples")
    
    random.shuffle(samples)
    print(f"\n✅ Loaded {len(samples)} total samples")
    return samples


# =============================================================================
# Clustering
# =============================================================================

def test_clustering_algorithms(embeddings: np.ndarray) -> Dict:
    """Test KMeans and HDBSCAN to find optimal clustering."""
    print("\n🔬 Testing clustering algorithms...")
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import hdbscan
    
    results = []
    
    # Test KMeans with different K
    print("\n   KMeans:")
    for k in range(3, 13):
        kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        sil = silhouette_score(embeddings, labels, metric='cosine')
        results.append({
            'algorithm': 'KMeans',
            'k': k,
            'silhouette': sil,
            'params': f'K={k}',
            'labels': labels,
            'centers': kmeans.cluster_centers_
        })
        print(f"      K={k:2d}: silhouette={sil:.4f}")
    
    # Test HDBSCAN with different parameters
    print("\n   HDBSCAN:")
    for min_cluster_size in [20, 30, 50]:
        for min_samples in [5, 10, 15]:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean'
            )
            labels = clusterer.fit_predict(embeddings)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            
            if n_clusters >= 2 and (labels != -1).sum() > n_clusters:
                mask = labels != -1
                sil = silhouette_score(embeddings[mask], labels[mask], metric='cosine')
            else:
                sil = -1
            
            # Compute centers manually
            unique_labels = sorted(set(labels) - {-1})
            centers = np.array([embeddings[labels == i].mean(axis=0) for i in unique_labels]) if unique_labels else None
            
            results.append({
                'algorithm': 'HDBSCAN',
                'k': n_clusters,
                'silhouette': sil,
                'params': f'mcs={min_cluster_size}, ms={min_samples}',
                'labels': labels,
                'centers': centers,
                'noise': n_noise
            })
            print(f"      mcs={min_cluster_size}, ms={min_samples}: K={n_clusters}, noise={n_noise}, sil={sil:.4f}")
    
    # Find best
    valid_results = [r for r in results if r['silhouette'] > 0]
    best = max(valid_results, key=lambda x: x['silhouette'])
    
    print(f"\n🏆 Best: {best['algorithm']} ({best['params']}) - silhouette={best['silhouette']:.4f}")
    
    return best


# =============================================================================
# Error Rate Simulation
# =============================================================================

def compute_error_rates(
    samples: List[Dict],
    labels: np.ndarray,
    models: List[Dict]
) -> Dict[str, List[float]]:
    """
    Compute per-cluster error rates for each model.
    
    Note: This simulates accuracy based on model size.
    For real accuracy, you'd run actual inference on each model.
    """
    print("\n📊 Computing per-cluster error rates...")
    
    unique_clusters = sorted(set(labels) - {-1})
    error_rates = {}
    
    for model in models:
        model_id = model['model_id']
        model_size = model['size_mb']
        
        rates = []
        for cluster_id in unique_clusters:
            cluster_mask = (labels == cluster_id)
            n_samples = cluster_mask.sum()
            
            if n_samples == 0:
                rates.append(0.5)
                continue
            
            # Simulate: larger models = lower error rate
            # Base error rate decreases with model size
            base_error = max(0.08, 0.60 - (model_size / 2000) * 0.50)
            
            # Add cluster-specific variation
            np.random.seed(hash(f"{model_id}_{cluster_id}") % 2**31)
            cluster_difficulty = 0.8 + 0.4 * np.random.random()
            
            error_rate = base_error * cluster_difficulty
            error_rate = np.clip(error_rate + np.random.uniform(-0.05, 0.05), 0.05, 0.70)
            
            rates.append(float(error_rate))
        
        error_rates[model_id] = rates
        avg_error = np.mean(rates)
        print(f"   {model_id:15s}: {avg_error:5.1%} avg error")
    
    return error_rates


# =============================================================================
# Profile Generation
# =============================================================================

def create_profile(
    embeddings: np.ndarray,
    best_clustering: Dict,
    error_rates: Dict[str, List[float]],
    models: List[Dict],
    embedding_dim: int
) -> Dict:
    """Create the final router profile."""
    
    unique_clusters = sorted(set(best_clustering['labels']) - {-1})
    
    profile = {
        'version': '2.0.0',
        'created_at': datetime.now().isoformat(),
        'metadata': {
            'embedding_source': 'cactus',  # Real Cactus embeddings!
            'embedding_model': 'LiquidAI/LFM2-350M',
            'embedding_dim': embedding_dim,
            'n_clusters': len(unique_clusters),
            'clustering_algorithm': best_clustering['algorithm'],
            'clustering_params': best_clustering['params'],
            'silhouette_score': float(best_clustering['silhouette']),
            'dataset': 'mmlu',
            'n_samples': len(embeddings),
            'topics': TOPICS,
        },
        'cluster_centers': best_clustering['centers'].tolist(),
        'llm_profiles': error_rates,
        'models': models,
    }
    
    return profile


# =============================================================================
# Main Training Function
# =============================================================================

def train():
    """Main training function."""
    print("=" * 60)
    print("🌵 AuroraAI Router - Cactus Integration Training")
    print("=" * 60)
    
    # 1. Initialize Cactus embedder
    print("\n📡 Initializing Cactus embedder...")
    embedder = CactusEmbedder(
        model_path=MODEL_PATH,
        lib_path=LIB_PATH
    )
    
    # 2. Load MMLU samples
    samples = load_mmlu_samples()
    
    # 3. Extract embeddings with Cactus
    print("\n🔢 Extracting embeddings with Cactus SDK...")
    texts = [s["question"] for s in samples]
    embeddings = embedder.embed_batch(texts, show_progress=True)
    print(f"   Shape: {embeddings.shape}")
    
    # 4. Find optimal clustering
    best_clustering = test_clustering_algorithms(embeddings)
    
    # 5. Compute error rates
    error_rates = compute_error_rates(
        samples,
        best_clustering['labels'],
        CACTUS_MODELS
    )
    
    # 6. Create profile
    print("\n📦 Creating profile...")
    profile = create_profile(
        embeddings,
        best_clustering,
        error_rates,
        CACTUS_MODELS,
        embedder.embedding_dim
    )
    
    # 7. Save profile
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = f"{OUTPUT_DIR}/cactus_router_profile.json"
    
    with open(output_path, 'w') as f:
        json.dump(profile, f, indent=2)
    
    file_size = os.path.getsize(output_path) / 1024
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"\n📊 Profile Statistics:")
    print(f"   File: {output_path}")
    print(f"   Size: {file_size:.1f} KB")
    print(f"   Embedding source: CACTUS (production-ready!)")
    print(f"   Embedding dim: {embedder.embedding_dim}")
    print(f"   Clusters: {profile['metadata']['n_clusters']}")
    print(f"   Algorithm: {best_clustering['algorithm']}")
    print(f"   Silhouette: {best_clustering['silhouette']:.4f}")
    print(f"   Models: {len(CACTUS_MODELS)}")
    
    print("\n📥 To download to your PC:")
    print(f"   scp ubuntu@<your-aws-ip>:{output_path} .")
    
    return profile


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    train()
