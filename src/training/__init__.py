"""
Training Module for AuroraAI Router.

Contains scripts for training router profiles:
- V1: KMeans clustering with MiniLM embeddings
- V2: HDBSCAN clustering with Nomic embeddings (CACTUS)
"""

from .train_v1 import TrainerV1
from .train_v2 import CactusTrainer

__all__ = ['TrainerV1', 'CactusTrainer']
