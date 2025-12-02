"""
Test script for Cactus Python bindings.

This script tests the ctypes bindings to ensure they work correctly.

Usage:
    # Check if Cactus is available
    python bindings/test_bindings.py

    # Test with a real model (on Mac)
    python bindings/test_bindings.py --model models/lfm2-350m-q8.gguf
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bindings import CactusModel, CactusError, CactusNotAvailableError, is_cactus_available


def test_availability():
    """Test if Cactus library is available."""
    print("=" * 60)
    print("Testing Cactus Availability")
    print("=" * 60)

    available, message = is_cactus_available()

    if available:
        print(f"✅ Cactus is available!")
        print(f"   {message}")
        return True
    else:
        print(f"❌ Cactus is not available")
        print(f"   {message}")
        return False


def test_model(model_path: str):
    """Test loading a model and generating embeddings."""
    print("\n" + "=" * 60)
    print("Testing Model Loading & Embeddings")
    print("=" * 60)

    try:
        # Initialize model
        print(f"\n1️⃣ Loading model: {model_path}")
        with CactusModel(model_path, context_size=2048) as model:

            # Get embedding dimension
            print("\n2️⃣ Getting embedding dimension...")
            dim = model.get_embedding_dim()
            print(f"   Embedding dimension: {dim}")

            # Test single embedding
            print("\n3️⃣ Testing single embedding...")
            test_text = "What is the capital of France?"
            embedding = model.embed(test_text)
            print(f"   Input: '{test_text}'")
            print(f"   Output shape: {embedding.shape}")
            print(f"   Output dtype: {embedding.dtype}")
            print(f"   Sample values: {embedding[:5]}")

            # Test batch embeddings
            print("\n4️⃣ Testing batch embeddings...")
            test_texts = [
                "What is quantum physics?",
                "Explain machine learning",
                "How does photosynthesis work?",
            ]
            embeddings = model.embed_batch(test_texts, show_progress=False)
            print(f"   Input: {len(test_texts)} texts")
            print(f"   Output shape: {embeddings.shape}")

            # Verify all embeddings have same dimension
            assert embeddings.shape == (len(test_texts), dim), "Shape mismatch!"
            print(f"   ✅ All embeddings have correct shape")

        print("\n✅ All tests passed!")
        return True

    except CactusNotAvailableError as e:
        print(f"\n❌ Cactus not available: {e}")
        return False
    except CactusError as e:
        print(f"\n❌ Cactus error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Cactus Python bindings")
    parser.add_argument(
        "--model",
        type=str,
        help="Path to GGUF model file (required for full test)"
    )
    parser.add_argument(
        "--lib-path",
        type=str,
        help="Path to libcactus shared library (optional)"
    )

    args = parser.parse_args()

    # Test availability
    available = test_availability()

    if not available:
        print("\n" + "=" * 60)
        print("ℹ️  Cactus is not available on this machine")
        print("=" * 60)
        print("\nThis is expected if you're on x86 (Windows/Intel Mac).")
        print("To test the bindings:")
        print("  1. Transfer this project to your Mac (ARM)")
        print("  2. Build Cactus: cd cactus && ./apple/build.sh")
        print("  3. Download model: ./cli/cactus download LiquidAI/LFM2-350M")
        print("  4. Run: python bindings/test_bindings.py --model models/lfm2-350m-q8.gguf")
        sys.exit(1)

    # If model provided, test it
    if args.model:
        if not Path(args.model).exists():
            print(f"\n❌ Model file not found: {args.model}")
            sys.exit(1)

        success = test_model(args.model)
        sys.exit(0 if success else 1)
    else:
        print("\n" + "=" * 60)
        print("ℹ️  Cactus library is available!")
        print("=" * 60)
        print("\nTo test with a model, run:")
        print("  python bindings/test_bindings.py --model path/to/model.gguf")


if __name__ == "__main__":
    main()
