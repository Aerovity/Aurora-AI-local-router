#!/usr/bin/env python3
"""
Train AuroraAI Router profile using HuggingFace Nomic embeddings.
Tests multiple clustering algorithms and picks the best.
"""
import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import AutoModel, AutoTokenizer

# Training imports
from datasets import load_dataset
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan

SEED = 42
np.random.seed(SEED)

class NomicEmbedder:
    def __init__(self):
        print('Loading Nomic embedding model...')
        self.model = AutoModel.from_pretrained(
            'nomic-ai/nomic-embed-text-v1.5', 
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            'nomic-ai/nomic-embed-text-v1.5'
        )
        self.model.eval()
        print(f'Model loaded! Embedding dim: 768')
    
    def embed_batch(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            prefixed = [f'search_query: {t}' for t in batch]
            inputs = self.tokenizer(prefixed, return_tensors='pt', padding=True, 
                                   truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                mask = inputs['attention_mask'].unsqueeze(-1)
                emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            all_embeddings.append(emb.numpy())
            if (i // batch_size) % 25 == 0:
                print(f'  Embedded {i + len(batch)}/{len(texts)} samples...')
        return np.vstack(all_embeddings)

def load_training_data(n_samples: int = 2000) -> List[Dict[str, Any]]:
    print(f'Loading MMLU dataset ({n_samples} samples)...')
    dataset = load_dataset('cais/mmlu', 'all', split='test', streaming=True)
    
    samples = []
    for item in dataset:
        if len(samples) >= n_samples:
            break
        samples.append({
            'text': item['question'],
            'subject': item['subject'],
            'choices': item['choices']
        })
        if len(samples) % 500 == 0:
            print(f'  Loaded {len(samples)} samples...')
    
    subjects = list(set(s["subject"] for s in samples))
    print(f'Loaded {len(samples)} samples from {len(subjects)} subjects')
    return samples, subjects

def test_all_clustering(embeddings: np.ndarray) -> Dict[str, Any]:
    """Test KMeans and HDBSCAN with various parameters, pick the best."""
    print('\n' + '='*60)
    print('🔬 Testing Clustering Algorithms')
    print('='*60)
    
    results = []
    normalized = normalize(embeddings)
    
    # Test KMeans with different K values
    print('\n📊 Testing KMeans:')
    for k in range(3, 12):
        try:
            kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10)
            labels = kmeans.fit_predict(normalized)
            sil = silhouette_score(normalized, labels, metric='cosine')
            results.append({
                'algo': 'KMeans',
                'k': k,
                'silhouette': sil,
                'params': f'k={k}',
                'noise': 0,
                'labels': labels,
                'centroids': kmeans.cluster_centers_
            })
            print(f'  K={k:2d}: silhouette={sil:.4f}')
        except Exception as e:
            print(f'  K={k:2d}: FAILED - {e}')
    
    # Test HDBSCAN with different parameters
    print('\n📊 Testing HDBSCAN:')
    for min_cluster_size in [15, 20, 30, 50, 75, 100]:
        for min_samples in [3, 5, 10, 15]:
            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric='euclidean',
                    cluster_selection_method='eom'
                )
                labels = clusterer.fit_predict(normalized)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = (labels == -1).sum()
                
                # Need at least 2 clusters and non-noise samples
                if n_clusters >= 2 and (labels != -1).sum() > n_clusters:
                    mask = labels != -1
                    sil = silhouette_score(normalized[mask], labels[mask], metric='cosine')
                    
                    # Compute centroids for non-noise clusters
                    unique_labels = sorted(set(labels) - {-1})
                    centroids = np.array([normalized[labels == i].mean(axis=0) for i in unique_labels])
                    
                    results.append({
                        'algo': 'HDBSCAN',
                        'k': n_clusters,
                        'silhouette': sil,
                        'params': f'mcs={min_cluster_size}, ms={min_samples}',
                        'noise': n_noise,
                        'noise_ratio': n_noise / len(labels),
                        'labels': labels,
                        'centroids': centroids
                    })
                    print(f'  mcs={min_cluster_size:3d}, ms={min_samples:2d}: K={n_clusters:2d}, noise={n_noise:4d} ({100*n_noise/len(labels):.1f}%), sil={sil:.4f}')
                else:
                    print(f'  mcs={min_cluster_size:3d}, ms={min_samples:2d}: K={n_clusters:2d} (too few clusters or too much noise)')
            except Exception as e:
                print(f'  mcs={min_cluster_size:3d}, ms={min_samples:2d}: FAILED - {e}')
    
    if not results:
        raise RuntimeError("No valid clustering found!")
    
    # Find best by silhouette score
    best = max(results, key=lambda x: x['silhouette'])
    
    print('\n' + '='*60)
    print(f'🏆 BEST: {best["algo"]} {best["params"]}')
    print(f'   Clusters: {best["k"]}')
    print(f'   Silhouette: {best["silhouette"]:.4f}')
    if best['algo'] == 'HDBSCAN':
        print(f'   Noise: {best["noise"]} ({100*best.get("noise_ratio", 0):.1f}%)')
    print('='*60)
    
    return best

def create_profile(best_clustering: Dict[str, Any], subjects: List[str], embedding_dim: int = 768) -> Dict[str, Any]:
    """Create production router profile."""
    labels = best_clustering['labels']
    centroids = best_clustering['centroids']
    
    # Get unique clusters (exclude noise for HDBSCAN)
    if best_clustering['algo'] == 'HDBSCAN':
        unique_labels = sorted(set(labels) - {-1})
    else:
        unique_labels = sorted(set(labels))
    
    # Build cluster centers dict
    cluster_centers = {}
    cluster_sizes = {}
    for i, label in enumerate(unique_labels):
        cluster_centers[str(label)] = centroids[i].tolist()
        cluster_sizes[str(label)] = int((labels == label).sum())
    
    # Assign models based on cluster size
    # Larger clusters = more common queries = use smaller/faster models
    cluster_models = {}
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: -x[1])
    n = len(sorted_clusters)
    for i, (cluster_id, size) in enumerate(sorted_clusters):
        if i < n // 3:
            cluster_models[cluster_id] = 'small'
        elif i < 2 * n // 3:
            cluster_models[cluster_id] = 'medium'
        else:
            cluster_models[cluster_id] = 'large'
    
    # Compute noise ratio
    noise_count = (labels == -1).sum() if -1 in labels else 0
    noise_ratio = noise_count / len(labels)
    
    profile = {
        'version': '1.0',
        'metadata': {
            'embedding_model': 'nomic-embed-text-v1.5',
            'embedding_dim': embedding_dim,
            'n_clusters': len(unique_labels),
            'clustering_algorithm': best_clustering['algo'],
            'clustering_params': best_clustering['params'],
            'silhouette_score': float(best_clustering['silhouette']),
            'noise_ratio': float(noise_ratio),
            'n_samples': len(labels),
            'subjects': subjects,
        },
        'cluster_centers': cluster_centers,
        'cluster_sizes': cluster_sizes,
        'cluster_models': cluster_models,
        'default_model': 'medium'
    }
    
    return profile

def main():
    parser = argparse.ArgumentParser(description='Train AuroraAI Router profile')
    parser.add_argument('--n-samples', type=int, default=2000, help='Number of training samples')
    parser.add_argument('--batch-size', type=int, default=8, help='Embedding batch size')
    parser.add_argument('--output', type=str, default='profiles/nomic_router_profile.json', help='Output file')
    args = parser.parse_args()
    
    # Initialize embedder
    embedder = NomicEmbedder()
    
    # Load data
    samples, subjects = load_training_data(args.n_samples)
    texts = [s['text'] for s in samples]
    
    # Generate embeddings
    print(f'\nGenerating embeddings (batch_size={args.batch_size})...')
    embeddings = embedder.embed_batch(texts, args.batch_size)
    print(f'Generated {len(embeddings)} embeddings of dim {embeddings.shape[1]}')
    
    # Test all clustering algorithms and pick best
    best_clustering = test_all_clustering(embeddings)
    
    # Create profile
    profile = create_profile(best_clustering, subjects)
    
    # Save profile
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(profile, f, indent=2)
    
    print(f'\n✅ Profile saved to {output_path}')
    print(f'   Algorithm: {profile["metadata"]["clustering_algorithm"]}')
    print(f'   Params: {profile["metadata"]["clustering_params"]}')
    print(f'   Clusters: {profile["metadata"]["n_clusters"]}')
    print(f'   Silhouette: {profile["metadata"]["silhouette_score"]:.4f}')
    print(f'   Noise ratio: {profile["metadata"]["noise_ratio"]:.2%}')
    print(f'   Embedding dim: {profile["metadata"]["embedding_dim"]}')

if __name__ == '__main__':
    main()
