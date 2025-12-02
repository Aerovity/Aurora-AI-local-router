"""
V2 Router - CACTUS Implementation

Uses Nomic embeddings with HDBSCAN clustering.
Better accuracy, trained on MMLU with real error rates.
"""

from .router import CactusRouter, RoutingResultV2
from .embedder import NomicEmbedder
from .cluster_engine import ClusterEngine

__all__ = ['CactusRouter', 'RoutingResultV2', 'NomicEmbedder', 'ClusterEngine']
