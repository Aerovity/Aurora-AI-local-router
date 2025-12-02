#!/usr/bin/env python3
"""
Test the Nomic KMeans router profile with HuggingFace embeddings.
This simulates what the mobile app will do with Cactus embeddings.

Usage:
    pip install torch transformers numpy
    python test_nomic_profile.py
"""

import json
import numpy as np
from pathlib import Path

def load_profile(profile_path: str) -> dict:
    """Load router profile from JSON file."""
    with open(profile_path, 'r') as f:
        return json.load(f)

def get_embedder():
    """Load Nomic embedding model (same as Cactus uses)."""
    print("Loading Nomic embedding model...")
    import torch
    from transformers import AutoModel, AutoTokenizer
    
    model = AutoModel.from_pretrained(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
    model.eval()
    
    def embed(text: str) -> np.ndarray:
        """Generate embedding for text."""
        # Nomic uses task prefixes
        prefixed = f"search_query: {text}"
        inputs = tokenizer(prefixed, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling with attention mask
            mask = inputs['attention_mask'].unsqueeze(-1)
            emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            # Normalize
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.numpy().flatten()
    
    print(f"Model loaded! Embedding dim: {model.config.hidden_size}")
    return embed

def find_nearest_cluster(embedding: np.ndarray, cluster_centers: dict) -> tuple:
    """Find nearest cluster using cosine similarity."""
    best_cluster = None
    best_similarity = -1
    
    for cluster_id, center in cluster_centers.items():
        center = np.array(center)
        # Cosine similarity
        similarity = np.dot(embedding, center) / (np.linalg.norm(embedding) * np.linalg.norm(center))
        if similarity > best_similarity:
            best_similarity = similarity
            best_cluster = cluster_id
    
    return best_cluster, best_similarity

def route_query(query: str, embed_fn, profile: dict) -> dict:
    """Route a query to the best model."""
    # Generate embedding
    embedding = embed_fn(query)
    
    # Find nearest cluster
    cluster_id, similarity = find_nearest_cluster(embedding, profile['cluster_centers'])
    
    # Get model assignment
    assignment = profile.get('model_assignments', {}).get(cluster_id, {})
    
    return {
        'query': query[:50] + '...' if len(query) > 50 else query,
        'cluster_id': cluster_id,
        'similarity': float(similarity),
        'recommended_model': assignment.get('model', 'unknown'),
        'confidence': assignment.get('confidence', 0.0)
    }

def main():
    # Find profile
    script_dir = Path(__file__).parent
    profile_paths = [
        script_dir / "profiles" / "production" / "nomic_kmeans_profile.json",
        script_dir.parent / "profiles" / "production" / "nomic_kmeans_profile.json",
        Path("profiles/production/nomic_kmeans_profile.json"),
    ]
    
    profile_path = None
    for p in profile_paths:
        if p.exists():
            profile_path = p
            break
    
    if not profile_path:
        print("ERROR: Could not find nomic_kmeans_profile.json")
        print("Looked in:", [str(p) for p in profile_paths])
        return
    
    print(f"Loading profile from: {profile_path}")
    profile = load_profile(profile_path)
    
    # Print profile info
    print("\n" + "="*60)
    print("PROFILE INFO")
    print("="*60)
    metadata = profile.get('metadata', {})
    print(f"  Embedding model: {metadata.get('embedding_model', 'unknown')}")
    print(f"  Embedding dim: {metadata.get('embedding_dim', 'unknown')}")
    print(f"  Clusters: {metadata.get('n_clusters', 'unknown')}")
    print(f"  Algorithm: {metadata.get('clustering_algorithm', 'unknown')}")
    print(f"  Silhouette: {metadata.get('silhouette_score', 0):.4f}")
    print(f"  Samples trained on: {metadata.get('n_samples', 'unknown')}")
    
    # Load embedder
    embed_fn = get_embedder()
    
    # Test queries
    test_queries = [
        # Simple queries (should route to smaller models)
        "What is 2 + 2?",
        "What color is the sky?",
        "How many days in a week?",
        
        # Medium complexity
        "Explain how photosynthesis works",
        "What are the main causes of World War I?",
        "How does machine learning differ from traditional programming?",
        
        # Complex queries (should route to larger models)
        "Derive the formula for gravitational potential energy from Newton's law of universal gravitation",
        "Compare and contrast the economic policies of Keynesian and Austrian schools of economics",
        "Explain the mathematical foundations of quantum entanglement and its implications for quantum computing",
    ]
    
    print("\n" + "="*60)
    print("ROUTING TEST RESULTS")
    print("="*60)
    
    for query in test_queries:
        result = route_query(query, embed_fn, profile)
        print(f"\nQuery: {result['query']}")
        print(f"  → Cluster: {result['cluster_id']}")
        print(f"  → Similarity: {result['similarity']:.4f}")
        print(f"  → Model: {result['recommended_model']}")
        print(f"  → Confidence: {result['confidence']:.2f}")
    
    # Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE MODE (type 'quit' to exit)")
    print("="*60)
    
    while True:
        try:
            query = input("\nEnter query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if not query:
                continue
            
            result = route_query(query, embed_fn, profile)
            print(f"  → Cluster: {result['cluster_id']}")
            print(f"  → Similarity: {result['similarity']:.4f}")
            print(f"  → Model: {result['recommended_model']}")
            
        except KeyboardInterrupt:
            break
    
    print("\nDone!")

if __name__ == "__main__":
    main()
