#!/usr/bin/env python3
"""
Test script to verify Cactus embeddings work correctly.
Run this before training to ensure everything is set up properly.
"""

import sys
import os
import time

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 50)
    print("  Cactus Embedding Test")
    print("=" * 50)
    print()
    
    # Check for model
    model_paths = [
        os.path.join(os.path.dirname(__file__), "..", "models", "lfm2-350m-q8.gguf"),
        os.path.expanduser("~/cactus-integration/models/lfm2-350m-q8.gguf"),
        "./models/lfm2-350m-q8.gguf",
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = os.path.abspath(path)
            break
    
    if model_path is None:
        print("ERROR: No model found!")
        print("Please run: ./scripts/download_model.sh")
        print()
        print("Or download manually from HuggingFace and place in models/")
        sys.exit(1)
    
    print(f"Model found: {model_path}")
    print()
    
    # Check for shared library
    lib_paths = [
        os.path.expanduser("~/cactus/build/cactus/libcactus.so"),
        "/usr/local/lib/libcactus.so",
        "./libcactus.so",
    ]
    
    lib_path = None
    for path in lib_paths:
        if os.path.exists(path):
            lib_path = path
            break
    
    if lib_path is None:
        print("ERROR: libcactus.so not found!")
        print("Please run: ./aws/build_shared_lib.sh")
        sys.exit(1)
    
    print(f"Library found: {lib_path}")
    print()
    
    # Try to load wrapper
    print("Loading Cactus wrapper...")
    try:
        from cactus_wrapper import CactusEmbedder
        print("✓ Wrapper loaded successfully")
    except ImportError as e:
        print(f"ERROR: Failed to import wrapper: {e}")
        sys.exit(1)
    
    # Initialize embedder
    print()
    print("Initializing Cactus embedder...")
    start = time.time()
    
    try:
        embedder = CactusEmbedder(model_path, lib_path)
        init_time = time.time() - start
        print(f"✓ Initialized in {init_time:.2f}s")
        print(f"  Embedding dimension: {embedder.get_embedding_dim()}")
    except Exception as e:
        print(f"ERROR: Failed to initialize: {e}")
        sys.exit(1)
    
    # Test single embedding
    print()
    print("Testing single embedding...")
    test_text = "What is the capital of France?"
    
    start = time.time()
    embedding = embedder.embed(test_text)
    embed_time = time.time() - start
    
    print(f"✓ Generated embedding in {embed_time:.3f}s")
    print(f"  Text: '{test_text}'")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  First 5 values: {embedding[:5]}")
    print(f"  Norm: {(embedding ** 2).sum() ** 0.5:.4f}")
    
    # Test batch embedding
    print()
    print("Testing batch embedding...")
    batch_texts = [
        "Solve for x: 2x + 5 = 13",
        "Write a poem about the ocean",
        "Explain quantum entanglement",
        "What are the symptoms of diabetes?",
        "How do I make pasta carbonara?",
    ]
    
    start = time.time()
    batch_embeddings = embedder.embed_batch(batch_texts)
    batch_time = time.time() - start
    
    print(f"✓ Generated {len(batch_texts)} embeddings in {batch_time:.3f}s")
    print(f"  Average: {batch_time/len(batch_texts)*1000:.1f}ms per embedding")
    print(f"  Batch shape: {batch_embeddings.shape}")
    
    # Test consistency
    print()
    print("Testing consistency...")
    embedding2 = embedder.embed(test_text)
    diff = ((embedding - embedding2) ** 2).sum() ** 0.5
    
    if diff < 1e-5:
        print(f"✓ Embeddings are deterministic (diff: {diff:.2e})")
    else:
        print(f"⚠ Embeddings vary slightly (diff: {diff:.2e})")
    
    # Cleanup
    embedder.close()
    
    print()
    print("=" * 50)
    print("  All tests passed! Ready to train.")
    print("=" * 50)
    print()
    print("Next step:")
    print("  python3 scripts/train_profile.py --model models/lfm2-350m-q8.gguf")


if __name__ == "__main__":
    main()
