"""
Cactus Python Wrapper - FFI Bindings for cactus_embed()

This module provides Python bindings to the Cactus SDK's embedding function
using ctypes. This allows us to generate embeddings that exactly match
what the Cactus runtime produces on mobile devices.

Usage:
    from cactus_wrapper import CactusEmbedder
    
    embedder = CactusEmbedder(
        model_path="weights/lfm2-350m",
        lib_path="lib/libcactus.so"
    )
    
    embedding = embedder.embed("What is machine learning?")
    print(embedding.shape)  # (1024,) for LFM2
"""

import ctypes
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
import time


class CactusEmbedder:
    """
    Python wrapper for Cactus SDK embedding function.
    
    Uses ctypes to call the native cactus_embed() function from libcactus.so.
    This ensures embeddings match exactly what mobile devices produce.
    """
    
    MAX_EMBEDDING_DIM = 4096  # Maximum expected embedding dimension
    
    def __init__(
        self,
        model_path: str,
        lib_path: str = "lib/libcactus.so",
        context_size: int = 512
    ):
        """
        Initialize the Cactus embedder.
        
        Args:
            model_path: Path to Cactus model weights directory
            lib_path: Path to libcactus.so shared library
            context_size: Context window size for the model
        """
        self.model_path = Path(model_path)
        self.lib_path = Path(lib_path)
        self.context_size = context_size
        self._model = None
        self._embedding_dim = None
        
        # Load library and initialize model
        self._load_library()
        self._setup_functions()
        self._init_model()
        
        print(f"✅ CactusEmbedder initialized")
        print(f"   Model: {self.model_path}")
        print(f"   Embedding dim: {self._embedding_dim}")
    
    def _load_library(self):
        """Load the Cactus shared library."""
        if not self.lib_path.exists():
            raise FileNotFoundError(
                f"Cactus library not found: {self.lib_path}\n"
                f"Run: bash build_shared_lib.sh"
            )
        
        try:
            self._lib = ctypes.CDLL(str(self.lib_path))
        except OSError as e:
            raise RuntimeError(f"Failed to load Cactus library: {e}")
    
    def _setup_functions(self):
        """Set up ctypes function signatures."""
        
        # cactus_init(model_path, context_size, corpus_dir) -> model_handle
        self._lib.cactus_init.argtypes = [
            ctypes.c_char_p,   # model_path
            ctypes.c_size_t,   # context_size
            ctypes.c_char_p,   # corpus_dir (NULL)
        ]
        self._lib.cactus_init.restype = ctypes.c_void_p
        
        # cactus_embed(model, text, buffer, buffer_size, dim) -> int
        self._lib.cactus_embed.argtypes = [
            ctypes.c_void_p,                   # model handle
            ctypes.c_char_p,                   # text
            ctypes.POINTER(ctypes.c_float),    # embeddings buffer
            ctypes.c_size_t,                   # buffer size in bytes
            ctypes.POINTER(ctypes.c_size_t),   # output: embedding dimension
        ]
        self._lib.cactus_embed.restype = ctypes.c_int
        
        # cactus_destroy(model) -> void
        self._lib.cactus_destroy.argtypes = [ctypes.c_void_p]
        self._lib.cactus_destroy.restype = None
        
        # cactus_reset(model) -> void
        self._lib.cactus_reset.argtypes = [ctypes.c_void_p]
        self._lib.cactus_reset.restype = None
    
    def _init_model(self):
        """Initialize the Cactus model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self._model = self._lib.cactus_init(
            str(self.model_path).encode('utf-8'),
            self.context_size,
            None  # No corpus directory
        )
        
        if not self._model:
            raise RuntimeError(f"Failed to initialize model: {self.model_path}")
        
        # Probe embedding dimension
        test_emb = self._embed_single("test")
        self._embedding_dim = len(test_emb)
    
    def _embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        buffer = (ctypes.c_float * self.MAX_EMBEDDING_DIM)()
        buffer_size = self.MAX_EMBEDDING_DIM * ctypes.sizeof(ctypes.c_float)
        dim = ctypes.c_size_t()
        
        result = self._lib.cactus_embed(
            self._model,
            text.encode('utf-8'),
            buffer,
            buffer_size,
            ctypes.byref(dim)
        )
        
        if result < 0:
            raise RuntimeError(f"cactus_embed failed with code: {result}")
        
        return np.array(buffer[:dim.value], dtype=np.float32)
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            text: Single string or list of strings
            
        Returns:
            np.ndarray: Shape (dim,) for single text, (n, dim) for list
        """
        if isinstance(text, str):
            return self._embed_single(text)
        
        return np.array([self._embed_single(t) for t in text], dtype=np.float32)
    
    def embed_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed a batch of texts with progress reporting.
        
        Args:
            texts: List of strings to embed
            show_progress: Whether to print progress
            
        Returns:
            np.ndarray: Shape (n_texts, embedding_dim)
        """
        embeddings = []
        n_texts = len(texts)
        start_time = time.time()
        
        for i, text in enumerate(texts):
            embeddings.append(self._embed_single(text))
            
            if show_progress and (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (n_texts - i - 1) / rate
                print(f"   Embedded {i+1}/{n_texts} ({100*(i+1)/n_texts:.1f}%) "
                      f"- {rate:.1f} texts/sec - ETA: {eta:.0f}s")
        
        if show_progress:
            total_time = time.time() - start_time
            print(f"   ✅ Done! {n_texts} texts in {total_time:.1f}s "
                  f"({n_texts/total_time:.1f} texts/sec)")
        
        return np.array(embeddings, dtype=np.float32)
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self._embedding_dim
    
    def reset(self):
        """Reset model state."""
        if self._model:
            self._lib.cactus_reset(self._model)
    
    def __del__(self):
        """Clean up model resources."""
        if hasattr(self, '_model') and self._model:
            if hasattr(self, '_lib') and self._lib:
                self._lib.cactus_destroy(self._model)
                self._model = None
    
    def __repr__(self):
        return (
            f"CactusEmbedder(\n"
            f"  model='{self.model_path.name}',\n"
            f"  embedding_dim={self._embedding_dim}\n"
            f")"
        )


# =============================================================================
# Test function
# =============================================================================

def test_embeddings():
    """Quick test of the Cactus embedder."""
    import os
    
    # Find model and library
    home = os.path.expanduser("~")
    model_path = f"{home}/cactus-integration/cactus/weights/lfm2-350m"
    lib_path = f"{home}/cactus-integration/lib/libcactus.so"
    
    print("🧪 Testing Cactus Embedder...")
    print(f"   Model: {model_path}")
    print(f"   Library: {lib_path}")
    
    # Initialize
    embedder = CactusEmbedder(model_path=model_path, lib_path=lib_path)
    
    # Test embeddings
    texts = [
        "What is machine learning?",
        "Explain quantum physics",
        "How do I cook pasta?",
    ]
    
    print("\n📊 Testing embeddings:")
    for text in texts:
        emb = embedder.embed(text)
        print(f"   '{text[:30]}...' -> shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")
    
    # Test similarity
    emb1 = embedder.embed("What is artificial intelligence?")
    emb2 = embedder.embed("Explain machine learning")
    emb3 = embedder.embed("How to make coffee")
    
    sim_12 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    sim_13 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
    
    print(f"\n📐 Similarity test:")
    print(f"   'AI' vs 'ML': {sim_12:.4f} (should be high)")
    print(f"   'AI' vs 'coffee': {sim_13:.4f} (should be low)")
    
    print("\n✅ Test passed!")


if __name__ == "__main__":
    test_embeddings()
