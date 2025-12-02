#!/usr/bin/env python3
"""
Memory-efficient KMeans training for t4g.small (1.8GB RAM).
Uses smaller batches and clears memory aggressively.
"""

import argparse
import json
import gc
import os
import numpy as np
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=1500)
    parser.add_argument('--output', type=str, default='nomic_kmeans_profile.json')
    args = parser.parse_args()
    
    print("=" * 60)
    print("MEMORY-EFFICIENT KMEANS TRAINING")
    print("=" * 60)
    
    # Step 1: Load data FIRST (before model)
    print("\n[1/4] Loading MMLU dataset...")
    from datasets import load_dataset
    
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
            ds = load_dataset("cais/mmlu", subject, split="test")
            for item in ds:
                if len(texts) >= args.n_samples:
                    break
                text = item['question']
                if item.get('choices'):
                    text += " " + " ".join(item['choices'][:4])
                texts.append(text)
        except:
            pass
        if len(texts) >= args.n_samples:
            break
    
    print(f"Loaded {len(texts)} samples")
    gc.collect()
    
    # Step 2: Generate embeddings in small batches
    print("\n[2/4] Generating embeddings (small batches)...")
    import torch
    from transformers import AutoModel, AutoTokenizer
    
    model = AutoModel.from_pretrained(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        safe_serialization=True
    )
    tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
    model.eval()
    
    embedding_dim = model.config.hidden_size
    print(f"Model loaded! Dim: {embedding_dim}")
    
    # Process in very small batches to save memory
    embeddings = []
    batch_size = 8  # Small batch
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch = ["search_document: " + t for t in batch]
            
            inputs = tokenizer(batch, padding=True, truncation=True, 
                             max_length=256, return_tensors="pt")  # Shorter max_length
            outputs = model(**inputs)
            
            # Mean pooling
            mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            summed = torch.sum(outputs.last_hidden_state * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            emb = summed / counts
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            
            embeddings.append(emb.numpy())
            
            # Clear GPU/CPU cache
            del inputs, outputs, mask, summed, counts, emb
            
            if (i // batch_size) % 20 == 0:
                print(f"  {min(i+batch_size, len(texts))}/{len(texts)}")
                gc.collect()
    
    embeddings = np.vstack(embeddings)
    print(f"Generated {len(embeddings)} embeddings")
    
    # Free model memory
    del model, tokenizer
    gc.collect()
    
    # Step 3: KMeans clustering
    print("\n[3/4] Running KMeans clustering...")
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    # Test K values
    best_k = 7
    best_score = -1
    best_kmeans = None
    
    for k in range(5, 12):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, metric='cosine', sample_size=min(1000, len(embeddings)))
        print(f"  K={k}: silhouette={score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            best_kmeans = kmeans
        
        gc.collect()
    
    print(f"\n🏆 Best: K={best_k}, silhouette={best_score:.4f}")
    
    # Step 4: Create profile
    print("\n[4/4] Creating profile...")
    
    labels = best_kmeans.labels_
    unique, counts = np.unique(labels, return_counts=True)
    
    cluster_centers = {}
    model_assignments = {}
    
    # Sort by cluster size (largest first = simpler queries)
    sorted_indices = np.argsort(-counts)
    
    model_tiers = ['local-tiny', 'local-small', 'local-medium', 
                   'local-large', 'cloud-small', 'cloud-medium',
                   'cloud-large', 'cloud-xl', 'cloud-premium',
                   'cloud-ultra', 'cloud-max']
    
    for rank, idx in enumerate(sorted_indices):
        cluster_id = unique[idx]
        cluster_centers[str(cluster_id)] = best_kmeans.cluster_centers_[cluster_id].tolist()
        model_idx = min(rank, len(model_tiers) - 1)
        model_assignments[str(cluster_id)] = {
            "model": model_tiers[model_idx],
            "confidence": round(0.90 - 0.02 * rank, 2),
            "cluster_size": int(counts[idx])
        }
        print(f"  Cluster {cluster_id}: {counts[idx]} samples -> {model_tiers[model_idx]}")
    
    profile = {
        "version": "1.0",
        "metadata": {
            "embedding_model": "nomic-embed-text-v1.5",
            "embedding_dim": embedding_dim,
            "n_clusters": best_k,
            "clustering_algorithm": "KMeans",
            "silhouette_score": float(best_score),
            "noise_ratio": 0.0,
            "n_samples": len(texts),
            "created_at": datetime.now().isoformat()
        },
        "cluster_centers": cluster_centers,
        "model_assignments": model_assignments
    }
    
    # Save
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(profile, f, indent=2)
    
    print(f"\n✅ Profile saved to {args.output}")
    print(f"   Clusters: {best_k}")
    print(f"   Silhouette: {best_score:.4f}")
    print(f"   Samples: {len(texts)}")

if __name__ == "__main__":
    main()
