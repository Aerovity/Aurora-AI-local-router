"""
V1 Embedder - MiniLM-based embeddings.

Lightweight embedding model suitable for mobile/edge deployment.
Uses sentence-transformers for easy integration.
"""

import numpy as np
from typing import List, Optional

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class MiniLMEmbedder:
    """MiniLM embedding model for V1 router.
    
    Uses all-MiniLM-L6-v2 (384 dimensions, ~80MB).
    Optimized for speed and low memory usage.
    """
    
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DIMENSIONS = 384
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize embedder.
        
        Args:
            model_name: Override model name (default: all-MiniLM-L6-v2)
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            
        self.model_name = model_name or self.DEFAULT_MODEL
        self.model = None
        self._loaded = False
        
    def load(self):
        """Load the embedding model."""
        if not self._loaded:
            print(f"  Loading MiniLM embeddings...")
            self.model = SentenceTransformer(self.model_name)
            self._loaded = True
            print(f"  ✅ MiniLM loaded ({self.DIMENSIONS} dims)")
            
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array of shape (384,)
        """
        if not self._loaded:
            self.load()
            
        embedding = self.model.encode(text, normalize_embeddings=False)
        return np.array(embedding, dtype=np.float32)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for batch of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            numpy array of shape (n_texts, 384)
        """
        if not self._loaded:
            self.load()
            
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            normalize_embeddings=False,
            show_progress_bar=len(texts) > 100
        )
        return np.array(embeddings, dtype=np.float32)
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
