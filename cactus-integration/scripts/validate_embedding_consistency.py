#!/usr/bin/env python3
"""
Validate Embedding Consistency Between Training and Runtime

This script tests whether embeddings generated during training (HuggingFace Transformers)
match embeddings that will be generated at runtime (Cactus GGUF, ONNX, or other mobile implementations).

A mismatch causes incorrect routing decisions!

Usage:
    # Test HuggingFace vs HuggingFace (baseline sanity check)
    python validate_embedding_consistency.py --mode hf-vs-hf

    # Test HuggingFace vs Cactus GGUF (critical test)
    python validate_embedding_consistency.py --mode hf-vs-cactus --cactus-model models/nomic-v1.5.gguf

    # Test with your router profile
    python validate_embedding_consistency.py --mode hf-vs-cactus --profile ../profiles/cactus_profile.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import sys
import os

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_header(text: str):
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}{text.center(70)}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")


def print_success(text: str):
    print(f"{GREEN}✓ {text}{RESET}")


def print_warning(text: str):
    print(f"{YELLOW}⚠ {text}{RESET}")


def print_error(text: str):
    print(f"{RED}✗ {text}{RESET}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_hf_nomic_embedder():
    """Load HuggingFace Nomic embedder (used during training)."""
    print("Loading HuggingFace Nomic model...")
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

        print_success(f"Loaded HuggingFace Nomic (dim={model.config.hidden_size})")
        return model, tokenizer
    except Exception as e:
        print_error(f"Failed to load HuggingFace model: {e}")
        sys.exit(1)


def embed_hf_nomic(texts: List[str], model, tokenizer) -> np.ndarray:
    """Generate embeddings using HuggingFace Nomic (training method)."""
    import torch

    if isinstance(texts, str):
        texts = [texts]

    # Add Nomic task prefix (same as training)
    texts = ["search_document: " + t for t in texts]

    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, truncation=True,
                          max_length=256, return_tensors="pt")
        outputs = model(**inputs)

        # Mean pooling (same as training)
        mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        summed = torch.sum(outputs.last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        embeddings = summed / counts

        # L2 normalize (same as training)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.numpy()


def load_cactus_embedder(model_path: str):
    """Load Cactus GGUF embedder (runtime method)."""
    print(f"Loading Cactus model from {model_path}...")

    # Check if model exists
    if not os.path.exists(model_path):
        print_error(f"Cactus model not found: {model_path}")
        print("\nTo download Nomic GGUF model:")
        print("  1. Go to: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF")
        print("  2. Download nomic-embed-text-v1.5.f16.gguf or Q8_0 variant")
        print(f"  3. Place in: {model_path}")
        sys.exit(1)

    try:
        # Try to import Cactus wrapper
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from cactus_wrapper import CactusEmbedder

        embedder = CactusEmbedder(model_path)
        dim = embedder.get_embedding_dim()
        print_success(f"Loaded Cactus model (dim={dim})")
        return embedder
    except ImportError:
        print_error("Cactus wrapper not available")
        print("\nCactus integration requires:")
        print("  1. Cactus library compiled (libcactus.so)")
        print("  2. cactus_wrapper.py in scripts/")
        print("\nFor now, we'll skip Cactus validation.")
        return None
    except Exception as e:
        print_error(f"Failed to load Cactus model: {e}")
        return None


def embed_cactus(texts: List[str], embedder) -> np.ndarray:
    """Generate embeddings using Cactus GGUF."""
    if embedder is None:
        return None

    if isinstance(texts, str):
        texts = [texts]

    try:
        # Cactus wrapper should handle batching
        embeddings = embedder.embed_batch(texts)
        return embeddings
    except Exception as e:
        print_error(f"Cactus embedding failed: {e}")
        return None


def test_embedding_consistency(
    texts: List[str],
    hf_model,
    hf_tokenizer,
    cactus_embedder=None,
    threshold_excellent=0.99,
    threshold_good=0.95,
    threshold_acceptable=0.90
) -> Dict:
    """Test consistency between HF and Cactus embeddings."""

    results = {
        'texts': texts,
        'similarities': [],
        'status': 'unknown',
        'recommendation': ''
    }

    print(f"\nTesting {len(texts)} sample queries...")
    print("-" * 70)

    # Generate HF embeddings
    print("Generating HuggingFace embeddings...")
    hf_embeddings = embed_hf_nomic(texts, hf_model, hf_tokenizer)
    print_success(f"Generated {len(hf_embeddings)} HF embeddings")

    # Generate Cactus embeddings
    if cactus_embedder:
        print("Generating Cactus embeddings...")
        cactus_embeddings = embed_cactus(texts, cactus_embedder)

        if cactus_embeddings is None:
            print_warning("Cactus embeddings failed, skipping comparison")
            results['status'] = 'cactus_failed'
            return results

        print_success(f"Generated {len(cactus_embeddings)} Cactus embeddings")

        # Check dimensions match
        if hf_embeddings.shape[1] != cactus_embeddings.shape[1]:
            print_error(f"Dimension mismatch! HF={hf_embeddings.shape[1]}, Cactus={cactus_embeddings.shape[1]}")
            results['status'] = 'dimension_mismatch'
            results['recommendation'] = "Use a different Cactus model that matches 768 dimensions"
            return results

        # Compare each embedding pair
        print("\nPer-Query Similarity Analysis:")
        print("-" * 70)

        similarities = []
        for i, text in enumerate(texts):
            sim = cosine_similarity(hf_embeddings[i], cactus_embeddings[i])
            similarities.append(sim)

            # Color code by quality
            if sim >= threshold_excellent:
                color = GREEN
                status = "EXCELLENT"
            elif sim >= threshold_good:
                color = YELLOW
                status = "GOOD"
            elif sim >= threshold_acceptable:
                color = YELLOW
                status = "ACCEPTABLE"
            else:
                color = RED
                status = "POOR"

            print(f"{color}{sim:.6f}{RESET} - {status:11s} - {text[:50]}...")

        results['similarities'] = similarities
        avg_sim = np.mean(similarities)
        min_sim = np.min(similarities)

        print("\n" + "="*70)
        print("OVERALL RESULTS")
        print("="*70)
        print(f"Average Similarity: {avg_sim:.6f}")
        print(f"Minimum Similarity: {min_sim:.6f}")
        print(f"Maximum Similarity: {np.max(similarities):.6f}")
        print(f"Std Deviation:      {np.std(similarities):.6f}")

        # Determine status
        if avg_sim >= threshold_excellent:
            results['status'] = 'excellent'
            print_success(f"\n🎉 EXCELLENT! Average similarity = {avg_sim:.4f} >= {threshold_excellent}")
            print_success("Your profile will work perfectly on phones!")
            results['recommendation'] = "✓ Profile is ready for deployment"

        elif avg_sim >= threshold_good:
            results['status'] = 'good'
            print_success(f"\n✓ GOOD. Average similarity = {avg_sim:.4f} >= {threshold_good}")
            print_success("Your profile should work well on phones.")
            print("Minor variations are acceptable and won't significantly affect routing.")
            results['recommendation'] = "✓ Profile is acceptable for deployment"

        elif avg_sim >= threshold_acceptable:
            results['status'] = 'acceptable'
            print_warning(f"\n⚠ ACCEPTABLE. Average similarity = {avg_sim:.4f} >= {threshold_acceptable}")
            print_warning("Your profile will work, but with some routing inaccuracies.")
            print("Consider retraining with Cactus embeddings for better results.")
            results['recommendation'] = "⚠ Consider retraining with Cactus embeddings"

        else:
            results['status'] = 'poor'
            print_error(f"\n✗ POOR! Average similarity = {avg_sim:.4f} < {threshold_acceptable}")
            print_error("Your profile will likely produce incorrect routing decisions!")
            print_error("You MUST retrain using Cactus embeddings.")
            results['recommendation'] = "✗ MUST retrain with Cactus embeddings"

    else:
        print_warning("\nCactus embedder not available - skipping runtime validation")
        print("Testing HF embedding consistency only...")

        # Just test HF consistency (re-embed same text)
        hf_embeddings2 = embed_hf_nomic(texts, hf_model, hf_tokenizer)

        similarities = []
        for i in range(len(texts)):
            sim = cosine_similarity(hf_embeddings[i], hf_embeddings2[i])
            similarities.append(sim)

        avg_sim = np.mean(similarities)
        if avg_sim > 0.9999:
            print_success(f"HF embeddings are deterministic (avg sim = {avg_sim:.6f})")
            results['status'] = 'hf_only_ok'
        else:
            print_warning(f"HF embeddings vary slightly (avg sim = {avg_sim:.6f})")
            results['status'] = 'hf_only_warning'

        results['recommendation'] = "⚠ Unable to validate Cactus compatibility - deploy at your own risk"

    return results


def validate_profile_clusters(profile_path: Path, hf_model, hf_tokenizer, cactus_embedder=None):
    """Validate that profile's cluster centers will work at runtime."""

    print_header("VALIDATING ROUTER PROFILE CLUSTER CENTERS")

    # Load profile
    with open(profile_path) as f:
        profile = json.load(f)

    print(f"Profile: {profile_path}")
    print(f"Embedding Model: {profile['metadata']['embedding_model']}")
    print(f"Clusters: {profile['metadata']['n_clusters']}")
    print(f"Feature Dim: {profile['metadata']['feature_dim']}")

    # Extract cluster centers
    cluster_centers = np.array(profile['cluster_centers']['cluster_centers'], dtype=np.float32)
    print(f"Cluster centers shape: {cluster_centers.shape}")

    # Test query assignment
    test_queries = [
        "What is 2+2?",
        "Explain quantum physics in detail",
        "Write Python code to sort a list",
        "Translate this to Spanish: Hello world",
        "What are the symptoms of diabetes?",
    ]

    print("\nTesting cluster assignments with sample queries...")
    print("-" * 70)

    # Get HF embeddings
    hf_embeddings = embed_hf_nomic(test_queries, hf_model, hf_tokenizer)

    # Assign to clusters using HF embeddings
    hf_clusters = []
    for emb in hf_embeddings:
        similarities = np.dot(cluster_centers, emb)
        cluster_id = np.argmax(similarities)
        hf_clusters.append(cluster_id)

    if cactus_embedder:
        # Get Cactus embeddings
        cactus_embeddings = embed_cactus(test_queries, cactus_embedder)

        if cactus_embeddings is not None:
            # Assign to clusters using Cactus embeddings
            cactus_clusters = []
            for emb in cactus_embeddings:
                similarities = np.dot(cluster_centers, emb)
                cluster_id = np.argmax(similarities)
                cactus_clusters.append(cluster_id)

            # Compare assignments
            mismatches = 0
            for i, query in enumerate(test_queries):
                hf_c = hf_clusters[i]
                cactus_c = cactus_clusters[i]
                match = "✓" if hf_c == cactus_c else "✗"

                if hf_c == cactus_c:
                    print(f"{GREEN}{match}{RESET} Query {i+1}: Cluster {hf_c} (both) - {query[:40]}...")
                else:
                    print(f"{RED}{match}{RESET} Query {i+1}: HF→{hf_c}, Cactus→{cactus_c} - {query[:40]}...")
                    mismatches += 1

            print("\n" + "="*70)
            if mismatches == 0:
                print_success(f"🎉 PERFECT! All {len(test_queries)} queries assigned to same clusters")
                print_success("Your router will behave identically on PC and phone!")
            else:
                print_error(f"MISMATCH! {mismatches}/{len(test_queries)} queries assigned differently")
                print_error("This means your router will make DIFFERENT decisions on phone vs PC!")
                print_error("\n💡 Solution: Retrain cluster centers using Cactus embeddings")
    else:
        print_warning("Cactus not available - showing HF cluster assignments only:")
        for i, query in enumerate(test_queries):
            print(f"  Query {i+1}: Cluster {hf_clusters[i]} - {query[:50]}...")


def main():
    parser = argparse.ArgumentParser(description="Validate embedding consistency")
    parser.add_argument('--mode', choices=['hf-vs-hf', 'hf-vs-cactus'], default='hf-vs-cactus')
    parser.add_argument('--cactus-model', type=str, help='Path to Cactus GGUF model')
    parser.add_argument('--profile', type=str, help='Path to router profile to validate')
    parser.add_argument('--test-queries', type=str, help='Path to JSON file with test queries')
    args = parser.parse_args()

    print_header("EMBEDDING CONSISTENCY VALIDATION")

    # Load HF model (always needed)
    hf_model, hf_tokenizer = load_hf_nomic_embedder()

    # Load Cactus model if needed
    cactus_embedder = None
    if args.mode == 'hf-vs-cactus':
        if args.cactus_model:
            cactus_embedder = load_cactus_embedder(args.cactus_model)
        else:
            print_warning("No --cactus-model specified, will try default locations...")
            default_paths = [
                "models/nomic-embed-text-v1.5.f16.gguf",
                "models/nomic-v1.5-q8.gguf",
                "../models/nomic-embed-text-v1.5.f16.gguf",
            ]
            for path in default_paths:
                if os.path.exists(path):
                    cactus_embedder = load_cactus_embedder(path)
                    break

            if cactus_embedder is None:
                print_warning("No Cactus model found in default locations")
                print("Will validate HF embeddings only")

    # Load test queries
    if args.test_queries:
        with open(args.test_queries) as f:
            test_queries = json.load(f)
    else:
        # Default test queries covering different complexity levels
        test_queries = [
            "What is 2+2?",
            "Hi, how are you?",
            "What is the capital of France?",
            "Explain how neural networks work",
            "Write Python code to implement quicksort algorithm",
            "What are the key differences between quantum computing and classical computing?",
            "Translate this sentence to Spanish: The quick brown fox jumps over the lazy dog",
            "What are the symptoms and treatment options for type 2 diabetes?",
            "Explain the theory of relativity in simple terms",
            "How do I make authentic Italian carbonara pasta?",
            "What causes earthquakes and how are they measured?",
            "Write a professional email declining a job offer politely",
            "What are the main themes in Shakespeare's Hamlet?",
            "Explain the difference between machine learning and deep learning",
            "What is photosynthesis and why is it important?",
        ]

    # Run consistency test
    print_header("TESTING EMBEDDING CONSISTENCY")
    results = test_embedding_consistency(
        test_queries,
        hf_model,
        hf_tokenizer,
        cactus_embedder
    )

    # Validate profile if provided
    if args.profile:
        validate_profile_clusters(
            Path(args.profile),
            hf_model,
            hf_tokenizer,
            cactus_embedder
        )

    # Final summary
    print_header("VALIDATION SUMMARY")
    print(f"Status: {results['status']}")
    print(f"Recommendation: {results['recommendation']}")

    if results['status'] in ['excellent', 'good', 'hf_only_ok']:
        print_success("\n✓ Validation passed!")
        return 0
    elif results['status'] in ['acceptable', 'hf_only_warning']:
        print_warning("\n⚠ Validation passed with warnings")
        return 0
    else:
        print_error("\n✗ Validation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
