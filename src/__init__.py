"""
AuroraAI Router - Intelligent Local LLM Routing

This package provides two router implementations:
- v1: Original router using sentence-transformers (MiniLM) embeddings with KMeans
- v2: CACTUS router using Nomic embeddings with HDBSCAN clustering

Usage:
    # V1 Router (lighter, faster embedding, ~80MB model)
    from src.v1 import RouterV1, MiniLMEmbedder
    
    # V2 CACTUS Router (better clustering, Nomic embeddings, 768-dim)  
    from src.v2 import CactusRouter, NomicEmbedder, ClusterEngine
    
    # Unified interface (auto-detects profile version)
    from src import create_router
    
    # Training
    from src.training import TrainerV1, CactusTrainer
    
    # Benchmarking
    from src.benchmarks import RouterBenchmark, compare_routers

CLI:
    python -m src train --version v2 --output profiles/my_profile.json
    python -m src benchmark --profile profiles/cactus_profile.json
    python -m src route --profile profiles/cactus_profile.json "What is quantum physics?"
    python -m src compare --v2 profiles/cactus_profile.json
"""

from .config import CACTUS_MODELS, MODEL_BY_ID, EmbeddingConfig, PathConfig
from .router_factory import create_router, RouterType, detect_profile_version

__version__ = "2.0.0"
__author__ = "AuroraAI Team"

__all__ = [
    # Factory
    'create_router',
    'RouterType',
    'detect_profile_version',
    
    # Configuration
    'CACTUS_MODELS',
    'MODEL_BY_ID',
    'EmbeddingConfig',
    'PathConfig',
]
