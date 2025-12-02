#!/usr/bin/env python3
"""
Train Cactus Router Profile - Matches COLAB_profiling.ipynb logic exactly.

This script:
1. Loads MMLU dataset (15 diverse topics, ~1500+ samples)
2. Extracts embeddings using Nomic (Cactus-compatible, 768-dim)
3. Tests BOTH KMeans AND HDBSCAN clustering
4. Simulates error rates for all 12 Cactus models per cluster
5. Saves production router profile matching COLAB format

Usage:
    python train_cactus_profile.py --n-samples 2000 --output cactus_profile.json
"""

import argparse
import json
import gc
import os
import random
import numpy as np
from datetime import datetime

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# All 12 Cactus models (from COLAB notebook)
CACTUS_MODELS = [
    {
        'model_id': 'gemma-270m',
        'model_path': 'google/gemma-3-270m-it',
        'size_mb': 172,
        'avg_tokens_per_sec': 173,
        'capabilities': ['text'],
        'context_size': 2048
    },
    {
        'model_id': 'lfm2-350m',
        'model_path': 'LiquidAI/LFM2-350M',
        'size_mb': 233,
        'avg_tokens_per_sec': 145,
        'capabilities': ['text', 'tools', 'embed'],
        'context_size': 2048
    },
    {
        'model_id': 'smollm-360m',
        'model_path': 'HuggingFaceTB/SmolLM2-360m-Instruct',
        'size_mb': 227,
        'avg_tokens_per_sec': 150,
        'capabilities': ['text'],
        'context_size': 2048
    },
    {
        'model_id': 'qwen-600m',
        'model_path': 'Qwen/Qwen3-0.6B',
        'size_mb': 394,
        'avg_tokens_per_sec': 129,
        'capabilities': ['text', 'tools', 'embed'],
        'context_size': 2048
    },
    {
        'model_id': 'lfm2-vl-450m',
        'model_path': 'LiquidAI/LFM2-VL-450M',
        'size_mb': 420,
        'avg_tokens_per_sec': 113,
        'capabilities': ['text', 'vision', 'embed'],
        'context_size': 2048
    },
    {
        'model_id': 'lfm2-700m',
        'model_path': 'LiquidAI/LFM2-700M',
        'size_mb': 467,
        'avg_tokens_per_sec': 115,
        'capabilities': ['text', 'tools', 'embed'],
        'context_size': 2048
    },
    {
        'model_id': 'gemma-1b',
        'model_path': 'google/gemma-3-1b-it',
        'size_mb': 642,
        'avg_tokens_per_sec': 100,
        'capabilities': ['text'],
        'context_size': 2048
    },
    {
        'model_id': 'lfm2-1.2b',
        'model_path': 'LiquidAI/LFM2-1.2B',
        'size_mb': 722,
        'avg_tokens_per_sec': 95,
        'capabilities': ['text', 'tools', 'embed'],
        'context_size': 2048
    },
    {
        'model_id': 'lfm2-1.2b-tools',
        'model_path': 'LiquidAI/LFM2-1.2B-Tools',
        'size_mb': 722,
        'avg_tokens_per_sec': 95,
        'capabilities': ['text', 'tools', 'embed'],
        'context_size': 2048
    },
    {
        'model_id': 'qwen-1.7b',
        'model_path': 'Qwen/Qwen3-1.7B',
        'size_mb': 1161,
        'avg_tokens_per_sec': 75,
        'capabilities': ['text', 'tools', 'embed'],
        'context_size': 2048
    },
    {
        'model_id': 'smollm-1.7b',
        'model_path': 'HuggingFaceTB/SmolLM2-1.7B-Instruct',
        'size_mb': 1161,
        'avg_tokens_per_sec': 72,
        'capabilities': ['text', 'embed'],
        'context_size': 2048
    },
    {
        'model_id': 'lfm2-vl-1.6b',
        'model_path': 'LiquidAI/LFM2-VL-1.6B',
        'size_mb': 1440,
        'avg_tokens_per_sec': 60,
        'capabilities': ['text', 'vision', 'embed'],
        'context_size': 2048
    },
]

# MMLU topics (same as COLAB)
TOPICS = [
    "abstract_algebra",
    "anatomy",
    "world_religions",
    "computer_security",
    "astronomy",
    "international_law",
    "marketing",
    "high_school_geography",
    "philosophy",
    "electrical_engineering",
    "high_school_physics",
    "econometrics",
    "moral_scenarios",
    "professional_medicine",
    "virology",
]


def simulate_cactus_inference(model_id, model_size_mb, question):
    """
    Simulate Cactus model performance based on model size.
    Matches COLAB notebook logic exactly.
    
    In production, replace with actual Cactus inference.
    """
    # Larger models = better accuracy (same formula as COLAB)
    base_accuracy = min(0.95, 0.40 + (model_size_mb / 1500) * 0.55)
    
    # Add deterministic randomness based on model+question
    np.random.seed(hash(model_id + question) % 2**32)
    is_correct = np.random.random() < base_accuracy
    
    return is_correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=2000)
    parser.add_argument('--output', type=str, default='cactus_nomic_profile.json')
    args = parser.parse_args()
    
    print("=" * 60)
    print("CACTUS ROUTER TRAINING (COLAB-COMPATIBLE)")
    print("=" * 60)
    print(f"Samples: {args.n_samples}")
    print(f"Output: {args.output}")
    print(f"Models: {len(CACTUS_MODELS)}")
    print(f"Topics: {len(TOPICS)}")
    
    # =========================================================
    # Step 1: Load MMLU Dataset
    # =========================================================
    print("\n[1/6] Loading MMLU dataset...")
    from datasets import load_dataset
    
    samples = []
    samples_per_topic = args.n_samples // len(TOPICS) + 10  # Extra to ensure enough
    
    for topic in TOPICS:
        try:
            ds = load_dataset("cais/mmlu", topic, split="test")
            topic_samples = list(ds)
            random.shuffle(topic_samples)
            
            for item in topic_samples[:samples_per_topic]:
                if len(samples) >= args.n_samples:
                    break
                samples.append({
                    'question': item['question'],
                    'subject': topic,
                    'choices': item['choices'],
                    'answer': item['answer']
                })
        except Exception as e:
            print(f"  Warning: {topic}: {e}")
        
        if len(samples) >= args.n_samples:
            break
    
    random.shuffle(samples)
    samples = samples[:args.n_samples]
    
    print(f"Loaded {len(samples)} samples from {len(TOPICS)} topics")
    
    # Show distribution
    topic_counts = {}
    for s in samples:
        topic_counts[s['subject']] = topic_counts.get(s['subject'], 0) + 1
    for topic, count in sorted(topic_counts.items()):
        print(f"  {topic:30s}: {count:3d}")
    
    gc.collect()
    
    # =========================================================
    # Step 2: Generate Nomic Embeddings
    # =========================================================
    print("\n[2/6] Generating Nomic embeddings...")
    import torch
    from transformers import AutoModel, AutoTokenizer
    
    EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
    
    model = AutoModel.from_pretrained(
        EMBEDDING_MODEL,
        trust_remote_code=True,
        safe_serialization=True
    )
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model.eval()
    
    embedding_dim = model.config.hidden_size
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"Embedding dim: {embedding_dim}")
    
    texts = [s['question'] for s in samples]
    embeddings = []
    batch_size = 8
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch = ["search_document: " + t for t in batch]
            
            inputs = tokenizer(batch, padding=True, truncation=True, 
                             max_length=256, return_tensors="pt")
            outputs = model(**inputs)
            
            # Mean pooling
            mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            summed = torch.sum(outputs.last_hidden_state * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            emb = summed / counts
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            
            embeddings.append(emb.numpy())
            
            del inputs, outputs, mask, summed, counts, emb
            
            if (i // batch_size) % 25 == 0:
                print(f"  {min(i+batch_size, len(texts))}/{len(texts)}")
                gc.collect()
    
    embeddings = np.vstack(embeddings)
    print(f"Generated {len(embeddings)} embeddings of dim {embeddings.shape[1]}")
    
    del model, tokenizer
    gc.collect()
    
    # =========================================================
    # Step 3: Test Clustering Algorithms (KMeans + HDBSCAN)
    # =========================================================
    print("\n[3/6] Testing clustering algorithms...")
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import hdbscan
    
    results = []
    
    # Test KMeans with K from 5 to 15 (same as COLAB)
    print("\n  Testing KMeans:")
    for k in range(5, 16):
        kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        sil = silhouette_score(embeddings, labels, metric='cosine')
        results.append({
            'algo': 'KMeans',
            'k': k,
            'silhouette': sil,
            'params': f'K={k}',
            'noise': 0
        })
        print(f"    K={k:2d}: silhouette={sil:.4f}")
        gc.collect()
    
    # Test HDBSCAN with different parameters (same as COLAB)
    print("\n  Testing HDBSCAN:")
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
            
            results.append({
                'algo': 'HDBSCAN',
                'k': n_clusters,
                'silhouette': sil,
                'params': f'mcs={min_cluster_size}, ms={min_samples}',
                'noise': n_noise
            })
            print(f"    mcs={min_cluster_size}, ms={min_samples:2d}: K={n_clusters:2d}, noise={n_noise:4d}, sil={sil:.4f}")
            gc.collect()
    
    # Find best configuration
    best = max(results, key=lambda x: x['silhouette'])
    print(f"\n🏆 BEST: {best['algo']} {best['params']}, K={best['k']}, Silhouette={best['silhouette']:.4f}")
    
    # =========================================================
    # Step 4: Apply Best Clustering
    # =========================================================
    print("\n[4/6] Applying best clustering...")
    
    if best['algo'] == 'KMeans':
        best_k = best['k']
        kmeans = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_
    else:
        # Parse HDBSCAN params
        params = best['params']
        mcs = int(params.split(',')[0].split('=')[1])
        ms = int(params.split(',')[1].split('=')[1])
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric='euclidean'
        )
        labels = clusterer.fit_predict(embeddings)
        
        # Compute centroids (exclude noise)
        unique_labels = sorted(set(labels) - {-1})
        centroids = np.array([embeddings[labels == i].mean(axis=0) for i in unique_labels])
    
    unique_clusters = sorted(set(labels) - {-1})
    
    print(f"Clusters: {len(unique_clusters)}")
    print(f"Centroids shape: {centroids.shape}")
    if -1 in labels:
        print(f"Noise samples: {(labels == -1).sum()}")
    
    # Show cluster sizes
    print("\nCluster sizes:")
    for c in unique_clusters:
        count = (labels == c).sum()
        print(f"  Cluster {c}: {count} samples")
    
    # =========================================================
    # Step 5: Compute Error Rates for All Models
    # =========================================================
    print("\n[5/6] Computing per-cluster error rates for each model...")
    
    error_rates = {}
    
    for model_info in CACTUS_MODELS:
        model_id = model_info['model_id']
        model_size = model_info['size_mb']
        
        rates = []
        
        for cluster_id in unique_clusters:
            # Get samples in this cluster
            cluster_mask = (labels == cluster_id)
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                rates.append(0.5)  # Default for empty
                continue
            
            # Run simulated inference
            correct_count = 0
            for idx in cluster_indices:
                question = samples[idx]['question']
                is_correct = simulate_cactus_inference(model_id, model_size, question)
                if is_correct:
                    correct_count += 1
            
            # Compute error rate
            accuracy = correct_count / len(cluster_indices)
            error_rate = 1.0 - accuracy
            rates.append(float(error_rate))
        
        error_rates[model_id] = rates
        avg_error = np.mean(rates)
        print(f"  {model_id:20s}: {avg_error:5.2%} avg error")
    
    print(f"\n✅ Error rates computed for {len(CACTUS_MODELS)} models across {len(unique_clusters)} clusters")
    
    # =========================================================
    # Step 6: Create & Save Profile (COLAB format)
    # =========================================================
    print("\n[6/6] Creating router profile...")
    
    profile = {
        'version': '1.0',
        'metadata': {
            'n_clusters': len(unique_clusters),
            'feature_dim': embeddings.shape[1],
            'embedding_model': EMBEDDING_MODEL,
            'lambda_min': 0.0,
            'lambda_max': 2.0,
            'default_cost_preference': 0.5,
            'silhouette_score': float(best['silhouette']),
            'clustering_algorithm': best['algo'],
            'clustering_params': best['params'],
            'target': 'cactus_compute',
            'dataset': 'mmlu',
            'n_samples': len(samples),
            'topics': TOPICS,
            'created_at': datetime.now().isoformat()
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
    
    # Save
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(profile, f, indent=2)
    
    file_size_kb = os.path.getsize(args.output) / 1024
    
    print(f"\n✅ Router profile saved!")
    print(f"\n📊 Profile Statistics:")
    print(f"   File: {args.output}")
    print(f"   Size: {file_size_kb:.1f} KB")
    print(f"   Models: {len(CACTUS_MODELS)}")
    print(f"   Clusters: {len(unique_clusters)}")
    print(f"   Samples: {len(samples)}")
    print(f"   Silhouette: {best['silhouette']:.4f}")
    print(f"   Algorithm: {best['algo']} ({best['params']})")
    print(f"\n📱 Ready for mobile deployment!")


if __name__ == "__main__":
    main()
