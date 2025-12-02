#!/usr/bin/env python3
"""
Train AuroraAI Router profile using HuggingFace Nomic embeddings.
These produce the SAME 768-dim embeddings as Cactus Nomic model.
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
import hdbscan

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
    
    def embed(self, text: str) -> np.ndarray:
        # Nomic uses task prefixes
        prefixed = f'search_query: {text}'
        inputs = self.tokenizer(prefixed, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            emb = outputs.last_hidden_state.mean(dim=1)
        return emb.numpy().flatten()
    
    def embed_batch(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            prefixed = [f'search_query: {t}' for t in batch]
            inputs = self.tokenizer(prefixed, return_tensors='pt', padding=True, 
                                   truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling with attention mask
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
    
    print(f'Loaded {len(samples)} samples from {len(set(s["subject"] for s in samples))} subjects')
    return samples

def train_clusters(embeddings: np.ndarray, min_cluster_size: int = 50) -> Dict[str, Any]:
    print(f'Training HDBSCAN clusters on {len(embeddings)} embeddings...')
    
    # Normalize embeddings
    normalized = normalize(embeddings)
    
    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=10,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(normalized)
    
    # Compute cluster centers
    unique_labels = set(labels) - {-1}  # Exclude noise
    cluster_centers = {}
    cluster_sizes = {}
    
    for label in unique_labels:
        mask = labels == label
        cluster_embeddings = normalized[mask]
        center = cluster_embeddings.mean(axis=0)
        center = center / np.linalg.norm(center)  # Normalize center
        cluster_centers[int(label)] = center.tolist()
        cluster_sizes[int(label)] = int(mask.sum())
    
    print(f'Found {len(cluster_centers)} clusters')
    for label, size in sorted(cluster_sizes.items(), key=lambda x: -x[1])[:5]:
        print(f'  Cluster {label}: {size} samples')
    
    noise_count = (labels == -1).sum()
    print(f'  Noise points: {noise_count}')
    
    return {
        'cluster_centers': cluster_centers,
        'cluster_sizes': cluster_sizes,
        'n_clusters': len(cluster_centers),
        'noise_ratio': float(noise_count) / len(labels)
    }

def create_profile(cluster_data: Dict[str, Any], embedding_dim: int = 768) -> Dict[str, Any]:
    # Map clusters to model recommendations
    n_clusters = cluster_data['n_clusters']
    cluster_centers = cluster_data['cluster_centers']
    
    # Create model assignments (simple round-robin for now)
    models = ['small', 'medium', 'large']
    cluster_models = {}
    
    sorted_clusters = sorted(cluster_data['cluster_sizes'].items(), key=lambda x: -x[1])
    for i, (cluster_id, size) in enumerate(sorted_clusters):
        # Larger clusters get smaller models (more common queries)
        if i < len(sorted_clusters) // 3:
            cluster_models[str(cluster_id)] = 'small'
        elif i < 2 * len(sorted_clusters) // 3:
            cluster_models[str(cluster_id)] = 'medium'
        else:
            cluster_models[str(cluster_id)] = 'large'
    
    profile = {
        'version': '1.0',
        'embedding_model': 'nomic-embed-text-v1.5',
        'embedding_dim': embedding_dim,
        'n_clusters': n_clusters,
        'cluster_centers': cluster_centers,
        'cluster_models': cluster_models,
        'cluster_sizes': cluster_data['cluster_sizes'],
        'noise_ratio': cluster_data['noise_ratio'],
        'default_model': 'medium'
    }
    
    return profile

def main():
    parser = argparse.ArgumentParser(description='Train AuroraAI Router profile')
    parser.add_argument('--n-samples', type=int, default=2000, help='Number of training samples')
    parser.add_argument('--batch-size', type=int, default=8, help='Embedding batch size')
    parser.add_argument('--min-cluster-size', type=int, default=50, help='Minimum cluster size')
    parser.add_argument('--output', type=str, default='profiles/router_profile.json', help='Output file')
    args = parser.parse_args()
    
    # Initialize embedder
    embedder = NomicEmbedder()
    
    # Load data
    samples = load_training_data(args.n_samples)
    texts = [s['text'] for s in samples]
    
    # Generate embeddings
    print(f'Generating embeddings (batch_size={args.batch_size})...')
    embeddings = embedder.embed_batch(texts, args.batch_size)
    print(f'Generated {len(embeddings)} embeddings of dim {embeddings.shape[1]}')
    
    # Train clusters
    cluster_data = train_clusters(embeddings, args.min_cluster_size)
    
    # Create profile
    profile = create_profile(cluster_data)
    
    # Save profile
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(profile, f, indent=2)
    
    print(f'\nProfile saved to {output_path}')
    print(f'  Clusters: {profile["n_clusters"]}')
    print(f'  Embedding dim: {profile["embedding_dim"]}')
    print(f'  Noise ratio: {profile["noise_ratio"]:.2%}')

if __name__ == '__main__':
    main()
