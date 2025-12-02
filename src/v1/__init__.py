"""
V1 Router - Original Implementation

Uses sentence-transformers (MiniLM) for embeddings.
Lighter and faster, suitable for mobile deployment.
"""

from .router import RouterV1
from .embedder import MiniLMEmbedder

__all__ = ['RouterV1', 'MiniLMEmbedder']
