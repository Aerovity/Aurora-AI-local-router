"""
V1 Router - Original Implementation with MiniLM embeddings.

This is the first version of the AuroraAI router, using:
- sentence-transformers/all-MiniLM-L6-v2 for embeddings (384 dims)
- KMeans or simple clustering
- Designed for lightweight mobile deployment
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np

from .embedder import MiniLMEmbedder
from ..config import CACTUS_MODELS, MODEL_BY_ID, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class RoutingResultV1:
    """Result of V1 routing decision."""
    model_id: str
    model_path: str
    size_mb: float
    tokens_per_sec: float
    score: float
    cluster_id: int
    distance: float
    error_rate: float
    alternatives: List[Tuple[str, float]] = None
    
    def to_dict(self) -> dict:
        return {
            'model_id': self.model_id,
            'model_path': self.model_path,
            'size_mb': self.size_mb,
            'tokens_per_sec': self.tokens_per_sec,
            'score': self.score,
            'cluster_id': self.cluster_id,
            'distance': self.distance,
            'error_rate': self.error_rate,
            'alternatives': self.alternatives or []
        }


class RouterV1:
    """V1 Router using MiniLM embeddings.
    
    Features:
    - Lightweight (~80MB embedding model)
    - Fast inference (~10ms routing)
    - Mobile-optimized
    - Uses sentence-transformers
    
    Example:
        >>> router = RouterV1.from_profile("profiles/v1/default_profile.json")
        >>> result = router.route("Explain quantum physics")
        >>> print(result.model_id)
    """
    
    VERSION = "1.0"
    
    def __init__(
        self,
        cluster_centers: np.ndarray,
        error_rates: Dict[str, List[float]],
        models: List[ModelConfig],
        embedder: Optional[MiniLMEmbedder] = None,
        lambda_min: float = 0.0,
        lambda_max: float = 2.0,
        default_cost_preference: float = 0.5
    ):
        """Initialize V1 Router.
        
        Args:
            cluster_centers: Array of cluster centroids (n_clusters, embedding_dim)
            error_rates: Dict mapping model_id to per-cluster error rates
            models: List of available models
            embedder: Optional embedder (creates new one if not provided)
            lambda_min: Minimum lambda for cost-quality tradeoff
            lambda_max: Maximum lambda for cost-quality tradeoff
            default_cost_preference: Default cost preference (0=fast, 1=quality)
        """
        self.cluster_centers = cluster_centers
        self.n_clusters = len(cluster_centers)
        self.error_rates = error_rates
        self.models = {m.model_id: m for m in models}
        self.embedder = embedder or MiniLMEmbedder()
        
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.default_cost_preference = default_cost_preference
        
        # Compute size normalization
        sizes = [m.size_mb for m in models]
        self.min_size = min(sizes)
        self.max_size = max(sizes)
        self.size_range = self.max_size - self.min_size
        
        logger.info(f"Initialized RouterV1 with {len(models)} models, {self.n_clusters} clusters")
        
    @classmethod
    def from_profile(
        cls,
        profile_path: Path,
        models: Optional[List[ModelConfig]] = None,
        embedder: Optional[MiniLMEmbedder] = None
    ) -> "RouterV1":
        """Load router from profile file.
        
        Args:
            profile_path: Path to JSON profile
            models: Optional list of models (uses default CACTUS_MODELS if not provided)
            embedder: Optional embedder instance
            
        Returns:
            Configured RouterV1 instance
        """
        profile_path = Path(profile_path)
        
        with open(profile_path, 'r') as f:
            profile = json.load(f)
            
        # Load cluster centers
        cluster_centers = np.array(
            profile['cluster_centers']['cluster_centers'],
            dtype=np.float32
        )
        
        # Load error rates
        error_rates = profile['llm_profiles']
        
        # Load metadata
        metadata = profile.get('metadata', {})
        
        # Use provided models or defaults
        if models is None:
            models = CACTUS_MODELS
            
        return cls(
            cluster_centers=cluster_centers,
            error_rates=error_rates,
            models=models,
            embedder=embedder,
            lambda_min=metadata.get('lambda_min', 0.0),
            lambda_max=metadata.get('lambda_max', 2.0),
            default_cost_preference=metadata.get('default_cost_preference', 0.5)
        )
    
    def route(
        self,
        prompt: str,
        cost_preference: Optional[float] = None,
        available_models: Optional[List[str]] = None,
        return_alternatives: bool = False
    ) -> RoutingResultV1:
        """Route a prompt to the optimal model.
        
        Args:
            prompt: Input text prompt
            cost_preference: 0.0=fast/small, 1.0=quality/large
            available_models: Optional list of model IDs to consider
            return_alternatives: Whether to return alternative suggestions
            
        Returns:
            RoutingResultV1 with selected model and metadata
        """
        start_time = time.time()
        
        # Get embedding
        embedding = self.embedder.embed(prompt)
        
        # Find nearest cluster
        cluster_id, distance = self._find_nearest_cluster(embedding)
        
        # Get models to consider
        models_to_score = self._filter_models(available_models)
        
        # Calculate lambda
        pref = cost_preference if cost_preference is not None else self.default_cost_preference
        lambda_param = self._calculate_lambda(pref)
        
        # Score models
        scored_models = []
        for model_id in models_to_score:
            model = self.models[model_id]
            error_rate = self.error_rates.get(model_id, [0.5] * self.n_clusters)[cluster_id]
            normalized_size = self._normalize_size(model.size_mb)
            
            score = error_rate + lambda_param * normalized_size
            scored_models.append((model_id, score, error_rate))
            
        # Sort by score
        scored_models.sort(key=lambda x: x[1])
        
        # Select best
        best_model_id, best_score, best_error = scored_models[0]
        best_model = self.models[best_model_id]
        
        # Prepare alternatives
        alternatives = None
        if return_alternatives and len(scored_models) > 1:
            alternatives = [(mid, score) for mid, score, _ in scored_models[1:4]]
            
        routing_time = (time.time() - start_time) * 1000
        logger.debug(f"Routed to {best_model_id} in {routing_time:.2f}ms")
        
        return RoutingResultV1(
            model_id=best_model_id,
            model_path=best_model.model_path,
            size_mb=best_model.size_mb,
            tokens_per_sec=best_model.avg_tokens_per_sec,
            score=best_score,
            cluster_id=cluster_id,
            distance=float(distance),
            error_rate=best_error,
            alternatives=alternatives
        )
    
    def _find_nearest_cluster(self, embedding: np.ndarray) -> Tuple[int, float]:
        """Find nearest cluster to embedding."""
        distances = np.linalg.norm(self.cluster_centers - embedding, axis=1)
        cluster_id = int(np.argmin(distances))
        distance = distances[cluster_id]
        return cluster_id, distance
    
    def _filter_models(self, available_models: Optional[List[str]]) -> List[str]:
        """Filter models based on availability."""
        if available_models is None:
            return list(self.models.keys())
        return [m for m in available_models if m in self.models]
    
    def _calculate_lambda(self, cost_preference: float) -> float:
        """Calculate lambda from cost preference."""
        return self.lambda_max - cost_preference * (self.lambda_max - self.lambda_min)
    
    def _normalize_size(self, size_mb: float) -> float:
        """Normalize model size to [0, 1]."""
        if self.size_range < 1e-6:
            return 0.0
        return (size_mb - self.min_size) / self.size_range
    
    def get_info(self) -> dict:
        """Get router information."""
        return {
            'version': self.VERSION,
            'n_clusters': self.n_clusters,
            'n_models': len(self.models),
            'embedding_model': self.embedder.model_name,
            'embedding_dims': MiniLMEmbedder.DIMENSIONS,
            'lambda_range': (self.lambda_min, self.lambda_max),
            'models': list(self.models.keys())
        }
