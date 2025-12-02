#!/usr/bin/env python3
"""
Train KMeans router profile with Nomic embeddings.
KMeans is better for routing (no noise - every query gets assigned).
"""

import argparse
import json
import os
import sys
import numpy as np
from datetime import datetime

# Fix Windows encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=2000)
    parser.add_argument('--n-clusters', type=int, default=7, help='Number of clusters (default: 7 based on HDBSCAN finding 3 natural clusters, using elbow method)')
    parser.add_argument('--output', type=str, default='nomic_kmeans_profile.json')
    args = parser.parse_args()
    
    print("=" * 60)
    print("NOMIC EMBEDDINGS KMEANS ROUTER TRAINING")
    print("=" * 60)
    
    # Import dependencies
    import torch
    from transformers import AutoModel, AutoTokenizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from datasets import load_dataset
    
    # Load Nomic model
    print("\nLoading Nomic embedding model...")
    model = AutoModel.from_pretrained(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        safe_serialization=True
    )
    tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
    model.eval()
    
    embedding_dim = model.config.hidden_size
    print(f"Model loaded! Embedding dim: {embedding_dim}")
    
    # Load MMLU dataset
    print(f"\nLoading {args.n_samples} MMLU samples...")
    subjects = [
        "college_computer_science", "college_medicine", "business_ethics",
        "college_mathematics", "college_physics", "conceptual_physics",
        "college_biology", "anatomy", "electrical_engineering", "astronomy",
        "abstract_algebra", "college_chemistry", "econometrics",
        "clinical_knowledge", "computer_security"
    ]
    
    texts = []
    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
            for item in ds:
                if len(texts) >= args.n_samples:
                    break
                text = item['question']
                if item.get('choices'):
                    choices = item['choices']
                    text += " Options: " + ", ".join(choices[:4])
                texts.append(text)
        except Exception as e:
            print(f"  Warning: {subject}: {e}")
        if len(texts) >= args.n_samples:
            break
    
    print(f"Loaded {len(texts)} samples")
    
    # Generate embeddings
    print(f"\nGenerating embeddings...")
    embeddings = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            # Add Nomic task prefix
            batch = ["search_document: " + t for t in batch]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            outputs = model(**inputs)
            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = torch.sum(token_embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = summed / counts
            # Normalize
            mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            embeddings.append(mean_pooled.numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {min(i+batch_size, len(texts))}/{len(texts)}")
    
    embeddings = np.vstack(embeddings)
    print(f"Generated {len(embeddings)} embeddings of dim {embeddings.shape[1]}")
    
    # Find optimal K using elbow method
    print(f"\nFinding optimal K using silhouette scores...")
    k_range = range(3, 12)
    results = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        silhouette = silhouette_score(embeddings, labels, metric='cosine')
        inertia = kmeans.inertia_
        results.append({
            'k': k,
            'silhouette': silhouette,
            'inertia': inertia
        })
        print(f"  K={k}: silhouette={silhouette:.4f}, inertia={inertia:.1f}")
    
    # Find best K by silhouette
    best = max(results, key=lambda x: x['silhouette'])
    print(f"\n🏆 Best K by silhouette: K={best['k']} (silhouette={best['silhouette']:.4f})")
    
    # Use the requested K or best K
    final_k = args.n_clusters if args.n_clusters != 7 else best['k']
    print(f"\nUsing K={final_k} for final model")
    
    # Train final model
    final_kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    final_labels = final_kmeans.fit_predict(embeddings)
    final_silhouette = silhouette_score(embeddings, final_labels, metric='cosine')
    
    # Analyze clusters
    print(f"\nCluster distribution:")
    unique, counts = np.unique(final_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} samples ({100*count/len(final_labels):.1f}%)")
    
    # Create profile
    cluster_centers = {}
    model_assignments = {}
    
    # Map clusters to models (simple heuristic based on complexity)
    # Sort clusters by their centroid magnitude as proxy for complexity
    center_magnitudes = [np.linalg.norm(final_kmeans.cluster_centers_[i]) for i in range(final_k)]
    sorted_indices = np.argsort(center_magnitudes)
    
    # Assign models: smallest magnitude = simplest queries = fastest model
    model_tiers = ['cactus-local-tiny', 'cactus-local-small', 'cactus-local-medium', 
                   'cactus-local-large', 'cactus-local-xl', 'cactus-cloud-small',
                   'cactus-cloud-medium', 'cactus-cloud-large', 'cactus-cloud-xl']
    
    for i, cluster_id in enumerate(sorted_indices):
        cluster_centers[str(cluster_id)] = final_kmeans.cluster_centers_[cluster_id].tolist()
        model_idx = min(i, len(model_tiers) - 1)
        model_assignments[str(cluster_id)] = {
            "model": model_tiers[model_idx],
            "confidence": 0.85 + 0.02 * (final_k - i),  # Higher confidence for simpler tiers
            "complexity_tier": i,
            "cluster_size": int(counts[cluster_id])
        }
    
    profile = {
        "version": "1.0",
        "metadata": {
            "embedding_model": "nomic-embed-text-v1.5",
            "embedding_dim": embedding_dim,
            "n_clusters": final_k,
            "clustering_algorithm": "KMeans",
            "clustering_params": f"n_clusters={final_k}, n_init=10",
            "silhouette_score": float(final_silhouette),
            "noise_ratio": 0.0,  # KMeans has no noise
            "n_samples": len(texts),
            "subjects": subjects,
            "created_at": datetime.now().isoformat()
        },
        "cluster_centers": cluster_centers,
        "model_assignments": model_assignments,
        "performance_estimates": {
            "simple": {"avg_latency_ms": 50, "error_rate": 0.02},
            "medium": {"avg_latency_ms": 150, "error_rate": 0.05},
            "complex": {"avg_latency_ms": 500, "error_rate": 0.10}
        }
    }
    
    # Save profile
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(profile, f, indent=2)
    
    print(f"\n✅ Profile saved to {args.output}")
    print(f"   Clusters: {final_k}")
    print(f"   Silhouette: {final_silhouette:.4f}")
    print(f"   All queries assigned (no noise)")

if __name__ == "__main__":
    main()
