#!/usr/bin/env python3
"""
Example: Basic Router Usage

Demonstrates how to use the AuroraAI Router for query routing.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import create_router
from src.config import PathConfig


def main():
    """Run basic router example."""
    
    print("=" * 60)
    print("AuroraAI Router - Basic Usage Example")
    print("=" * 60)
    
    # Load router with auto-detection
    profile_path = PathConfig.PROFILE_PATH
    print(f"\nLoading router from: {profile_path}")
    
    if not profile_path.exists():
        print(f"ERROR: Profile not found at {profile_path}")
        print("Please run training first or check the path.")
        return
    
    router = create_router(profile_path)
    
    # Get router info
    info = router.get_info()
    print(f"\nRouter Info:")
    print(f"  Version: {info.get('version', 'Unknown')}")
    print(f"  Embedding Model: {info.get('embedding_model', 'Unknown')}")
    print(f"  Number of Clusters: {info.get('n_clusters', 'Unknown')}")
    
    # Example queries
    test_queries = [
        # Simple queries - should route to smaller models
        "What is 2 + 2?",
        "What color is the sky?",
        "Define the word 'happy'",
        
        # Medium complexity - should route to mid-size models
        "Explain how photosynthesis works",
        "What are the main causes of World War I?",
        "How do computers store information?",
        
        # Complex queries - should route to larger models
        "Analyze the philosophical implications of artificial consciousness",
        "Explain quantum entanglement and its applications in computing",
        "Compare and contrast the economic theories of Keynes and Hayek",
    ]
    
    print("\n" + "=" * 60)
    print("Routing Test Queries")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        result = router.route(query, cost_preference=0.5)
        
        print(f"\n[{i}] Query: {query[:50]}...")
        print(f"    Selected Model: {result.selected_model}")
        print(f"    Cluster: {result.cluster_id}")
        print(f"    Confidence: {result.confidence:.2%}")
    
    # Demonstrate cost preference
    print("\n" + "=" * 60)
    print("Cost Preference Comparison")
    print("=" * 60)
    
    test_query = "Explain machine learning algorithms"
    
    for preference in [0.0, 0.5, 1.0]:
        result = router.route(test_query, cost_preference=preference)
        label = {0.0: "Speed", 0.5: "Balanced", 1.0: "Quality"}[preference]
        print(f"\n  {label} (λ={preference}):")
        print(f"    Model: {result.selected_model}")
        print(f"    Confidence: {result.confidence:.2%}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
