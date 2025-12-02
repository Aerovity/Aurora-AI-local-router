#!/usr/bin/env python3
"""
Quick validation script - no Cactus required!

This script:
1. Tests if your HuggingFace Nomic embeddings are deterministic
2. Validates your router profile can be loaded
3. Shows cluster assignments for sample queries
4. Estimates expected routing behavior

Run this FIRST before the full validation.
"""

import json
import numpy as np
import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

# Simple ASCII fallbacks for Windows
CHECK = 'OK' if sys.platform == 'win32' else '✓'
CROSS = 'X' if sys.platform == 'win32' else '✗'
WARN = '!' if sys.platform == 'win32' else '⚠'


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_nomic():
    """Load Nomic model."""
    print("Loading Nomic model...")
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
            safe_serialization=True
        )
        tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
        model.eval()
        print(f"{GREEN}{CHECK} Loaded (dim={model.config.hidden_size}){RESET}\n")
        return model, tokenizer
    except Exception as e:
        print(f"{RED}{CROSS} Failed: {e}{RESET}")
        sys.exit(1)


def embed_texts(texts, model, tokenizer):
    """Generate embeddings."""
    import torch

    if isinstance(texts, str):
        texts = [texts]

    texts_prefixed = ["search_document: " + t for t in texts]

    with torch.no_grad():
        inputs = tokenizer(texts_prefixed, padding=True, truncation=True,
                          max_length=256, return_tensors="pt")
        outputs = model(**inputs)

        mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        summed = torch.sum(outputs.last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        embeddings = summed / counts
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.numpy()


def test_determinism(model, tokenizer):
    """Test if embeddings are deterministic."""
    print("Testing embedding determinism...")

    test_text = "What is the capital of France?"

    emb1 = embed_texts(test_text, model, tokenizer)[0]
    emb2 = embed_texts(test_text, model, tokenizer)[0]

    diff = np.linalg.norm(emb1 - emb2)

    if diff < 1e-6:
        print(f"{GREEN}{CHECK} Embeddings are deterministic (diff={diff:.2e}){RESET}")
        return True
    else:
        print(f"{YELLOW}{WARN} Embeddings vary slightly (diff={diff:.2e}){RESET}")
        return False


def validate_profile(profile_path, model, tokenizer):
    """Validate router profile."""
    print(f"\nValidating profile: {profile_path}")

    # Load profile
    try:
        with open(profile_path) as f:
            profile = json.load(f)
        print(f"{GREEN}{CHECK} Profile loaded{RESET}")
    except Exception as e:
        print(f"{RED}{CROSS} Failed to load profile: {e}{RESET}")
        return False

    # Check structure
    required_keys = ['metadata', 'cluster_centers', 'llm_profiles', 'models']
    for key in required_keys:
        if key not in profile:
            print(f"{RED}{CROSS} Missing key: {key}{RESET}")
            return False

    print(f"{GREEN}{CHECK} Profile structure valid{RESET}")

    # Extract info
    metadata = profile['metadata']
    print(f"\n{BLUE}Profile Info:{RESET}")
    print(f"  Embedding model: {metadata['embedding_model']}")
    print(f"  Clusters: {metadata['n_clusters']}")
    print(f"  Feature dim: {metadata['feature_dim']}")
    print(f"  Algorithm: {metadata['clustering_algorithm']}")
    print(f"  Silhouette: {metadata['silhouette_score']:.4f}")
    print(f"  Samples: {metadata['n_samples']}")

    # Load cluster centers
    cluster_centers = np.array(
        profile['cluster_centers']['cluster_centers'],
        dtype=np.float32
    )
    print(f"  Cluster centers shape: {cluster_centers.shape}")

    # Validate dimensions match
    if metadata['feature_dim'] != 768:
        print(f"{RED}{CROSS} Feature dim should be 768 for Nomic v1.5, got {metadata['feature_dim']}{RESET}")
        return False

    if cluster_centers.shape[1] != 768:
        print(f"{RED}{CROSS} Cluster centers have wrong dimension: {cluster_centers.shape[1]}{RESET}")
        return False

    print(f"{GREEN}{CHECK} Dimensions correct{RESET}")

    # Test cluster assignments
    test_queries = [
        ("What is 2+2?", "simple"),
        ("Explain quantum physics", "complex"),
        ("Write Python quicksort", "coding"),
        ("Hi, how are you?", "chat"),
        ("Symptoms of diabetes?", "medical"),
    ]

    print(f"\n{BLUE}Testing Cluster Assignments:{RESET}")
    print("-" * 60)

    embeddings = embed_texts([q for q, _ in test_queries], model, tokenizer)

    cluster_usage = {i: 0 for i in range(metadata['n_clusters'])}

    for i, (query, category) in enumerate(test_queries):
        emb = embeddings[i]

        # Assign to cluster (cosine similarity)
        similarities = np.dot(cluster_centers, emb)
        cluster_id = int(np.argmax(similarities))
        confidence = float(similarities[cluster_id])

        cluster_usage[cluster_id] += 1

        # Get model recommendations from profile
        llm_profiles = profile['llm_profiles']

        # Find best model for this cluster
        model_scores = []
        for model_info in profile['models']:
            model_id = model_info['model_id']
            error_rate = llm_profiles[model_id][cluster_id]
            model_scores.append((model_id, error_rate))

        model_scores.sort(key=lambda x: x[1])
        best_model = model_scores[0][0]
        best_error = model_scores[0][1]

        print(f"  {category:10s} → Cluster {cluster_id} (conf={confidence:.3f})")
        print(f"              → Best model: {best_model} (error={best_error:.2%})")

    print(f"\n{BLUE}Cluster Usage:{RESET}")
    for cluster_id, count in cluster_usage.items():
        print(f"  Cluster {cluster_id}: {count} queries")

    # Check for noise in HDBSCAN
    if metadata['clustering_algorithm'] == 'HDBSCAN':
        noise_ratio = 1594 / 2000  # From your training output
        if noise_ratio > 0.5:
            print(f"\n{YELLOW}{WARN} WARNING: {noise_ratio:.1%} of training samples were marked as noise!{RESET}")
            print(f"{YELLOW}  This means most queries won't have strong cluster assignments.{RESET}")
            print(f"{YELLOW}  Consider retraining with KMeans instead.{RESET}")
        else:
            print(f"\n{GREEN}{CHECK} Low noise ratio: {noise_ratio:.1%}{RESET}")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=str,
                       default='../profiles/cactus_profile.json',
                       help='Path to router profile')
    args = parser.parse_args()

    print("="*70)
    print("QUICK VALIDATION (HuggingFace Only)".center(70))
    print("="*70)
    print()

    # Load model
    model, tokenizer = load_nomic()

    # Test determinism
    is_deterministic = test_determinism(model, tokenizer)

    # Validate profile
    if Path(args.profile).exists():
        profile_valid = validate_profile(args.profile, model, tokenizer)
    else:
        print(f"\n{RED}{CROSS} Profile not found: {args.profile}{RESET}")
        print(f"  Run train_cactus_profile.py first!")
        profile_valid = False

    # Summary
    print("\n" + "="*70)
    print("SUMMARY".center(70))
    print("="*70)

    if is_deterministic and profile_valid:
        print(f"{GREEN}{CHECK} Basic validation passed{RESET}")
        print(f"\n{BLUE}Next steps:{RESET}")
        print(f"  1. Run full validation with Cactus:")
        print(f"     python validate_embedding_consistency.py --profile {args.profile}")
        print(f"  2. Or deploy and test on actual phone")
        return 0
    else:
        print(f"{RED}{CROSS} Validation failed{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
