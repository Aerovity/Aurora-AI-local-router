"""
V2 CACTUS Router - Clustering for Adaptive Context-aware Task-based Unified Sampling.

This is the second version of the AuroraAI router, using:
- nomic-ai/nomic-embed-text-v1.5 for embeddings (768 dims)
- HDBSCAN clustering (trained on MMLU)
- Per-cluster error rates from real model evaluation
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .embedder import NomicEmbedder
from .cluster_engine import ClusterEngine
from ..config import CACTUS_MODELS, MODEL_BY_ID, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class RoutingResultV2:
    """Result of V2 CACTUS routing decision."""
    model_id: str
    model_path: str
    size_mb: float
    tokens_per_sec: float
    score: float
    cluster_id: int
    distance: float
    error_rate: float
    estimated_latency_ms: float
    routing_latency_ms: float = 0.0
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    
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
            'estimated_latency_ms': self.estimated_latency_ms,
            'routing_latency_ms': self.routing_latency_ms,
            'alternatives': self.alternatives
        }


class CactusRouter:
    """V2 CACTUS Router using Nomic embeddings and HDBSCAN clustering.
    
    CACTUS = Clustering for Adaptive Context-aware Task-based Unified Sampling
    
    Features:
    - Nomic embeddings (768 dims, high quality)
    - HDBSCAN clustering (trained on MMLU)
    - Real per-cluster error rates
    - Cost-quality tradeoff via lambda parameter
    - Matryoshka support for mobile (dimension reduction)
    
    Example:
        >>> router = CactusRouter.from_profile("profiles/v2/cactus_profile.json")
        >>> result = router.route("Explain quantum physics", cost_preference=0.8)
        >>> print(result.model_id)  # Best model for this query type
    """
    
    VERSION = "2.0"
    
    def __init__(
        self,
        cluster_engine: ClusterEngine,
        error_rates: Dict[str, List[float]],
        models: List[ModelConfig],
        embedder: Optional[NomicEmbedder] = None,
        lambda_min: float = 0.0,
        lambda_max: float = 2.0,
        default_cost_preference: float = 0.5,
        metadata: Optional[dict] = None
    ):
        """Initialize CACTUS Router.
        
        Args:
            cluster_engine: Fitted cluster engine with centroids
            error_rates: Dict mapping model_id to per-cluster error rates
            models: List of available models
            embedder: Optional Nomic embedder (creates new one if not provided)
            lambda_min: Minimum lambda for cost-quality tradeoff
            lambda_max: Maximum lambda for cost-quality tradeoff
            default_cost_preference: Default cost preference (0=fast, 1=quality)
            metadata: Optional profile metadata
        """
        self.cluster_engine = cluster_engine
        self.error_rates = error_rates
        self.models = {m.model_id: m for m in models}
        self.embedder = embedder or NomicEmbedder()
        
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.default_cost_preference = default_cost_preference
        self.metadata = metadata or {}
        
        # Compute size normalization
        sizes = [m.size_mb for m in models]
        self.min_size = min(sizes)
        self.max_size = max(sizes)
        self.size_range = self.max_size - self.min_size
        
        logger.info(
            f"Initialized CactusRouter V2 with {len(models)} models, "
            f"{cluster_engine.n_clusters} clusters"
        )
        
    @classmethod
    def from_profile(
        cls,
        profile_path: Path,
        models: Optional[List[ModelConfig]] = None,
        embedder: Optional[NomicEmbedder] = None,
        auto_load_embedder: bool = True
    ) -> "CactusRouter":
        """Load router from CACTUS profile file.
        
        Args:
            profile_path: Path to JSON profile
            models: Optional list of models (uses default CACTUS_MODELS if not provided)
            embedder: Optional embedder instance
            auto_load_embedder: Whether to auto-load embedder
            
        Returns:
            Configured CactusRouter instance
        """
        profile_path = Path(profile_path)
        
        with open(profile_path, 'r') as f:
            profile = json.load(f)
            
        # Create cluster engine from profile
        cluster_engine = ClusterEngine.from_profile(profile)
        
        # Load error rates
        error_rates = profile['llm_profiles']
        
        # Load metadata
        metadata = profile.get('metadata', {})
        
        # Use provided models or defaults
        if models is None:
            models = CACTUS_MODELS
            
        # Create embedder if needed
        if embedder is None and auto_load_embedder:
            embedding_model = metadata.get('embedding_model', NomicEmbedder.DEFAULT_MODEL)
            embedder = NomicEmbedder(model_name=embedding_model)
            
        return cls(
            cluster_engine=cluster_engine,
            error_rates=error_rates,
            models=models,
            embedder=embedder,
            lambda_min=metadata.get('lambda_min', 0.0),
            lambda_max=metadata.get('lambda_max', 2.0),
            default_cost_preference=metadata.get('default_cost_preference', 0.5),
            metadata=metadata
        )
    
    def route(
        self,
        prompt: str,
        cost_preference: Optional[float] = None,
        available_models: Optional[List[str]] = None,
        required_capabilities: Optional[List[str]] = None,
        return_alternatives: bool = False
    ) -> RoutingResultV2:
        """Route a prompt to the optimal model.
        
        Args:
            prompt: Input text prompt
            cost_preference: 0.0=fast/small, 1.0=quality/large
            available_models: Optional list of model IDs to consider
            required_capabilities: Required capabilities (e.g., ['vision'])
            return_alternatives: Whether to return alternative suggestions
            
        Returns:
            RoutingResultV2 with selected model and metadata
        """
        start_time = time.time()
        
        # Get embedding
        embedding = self.embedder.embed(prompt)
        
        return self.route_from_embedding(
            embedding=embedding,
            cost_preference=cost_preference,
            available_models=available_models,
            required_capabilities=required_capabilities,
            return_alternatives=return_alternatives,
            start_time=start_time
        )
    
    def route_from_embedding(
        self,
        embedding: np.ndarray,
        cost_preference: Optional[float] = None,
        available_models: Optional[List[str]] = None,
        required_capabilities: Optional[List[str]] = None,
        return_alternatives: bool = False,
        start_time: Optional[float] = None
    ) -> RoutingResultV2:
        """Route using pre-computed embedding.
        
        Args:
            embedding: Pre-computed embedding vector
            cost_preference: 0.0=fast/small, 1.0=quality/large
            available_models: Optional list of model IDs to consider
            required_capabilities: Required capabilities
            return_alternatives: Whether to return alternatives
            start_time: Optional start time for latency calculation
            
        Returns:
            RoutingResultV2 with selected model
        """
        if start_time is None:
            start_time = time.time()
            
        # Find nearest cluster
        cluster_id, distance = self.cluster_engine.assign_cluster(embedding)
        
        # Filter models
        models_to_score = self._filter_models(available_models, required_capabilities)
        
        if not models_to_score:
            raise ValueError(
                f"No models available matching criteria: "
                f"available={available_models}, capabilities={required_capabilities}"
            )
        
        # Calculate lambda
        pref = cost_preference if cost_preference is not None else self.default_cost_preference
        lambda_param = self._calculate_lambda(pref)
        
        # Score models
        scored_models = []
        for model_id in models_to_score:
            model = self.models[model_id]
            
            # Get error rate for this cluster
            error_rates = self.error_rates.get(model_id, [0.5] * self.cluster_engine.n_clusters)
            error_rate = error_rates[cluster_id] if cluster_id < len(error_rates) else 0.5
            
            normalized_size = self._normalize_size(model.size_mb)
            
            # Score = error_rate + lambda * normalized_size
            score = error_rate + lambda_param * normalized_size
            
            # Estimate inference latency (100 tokens output)
            estimated_latency = (100.0 / model.avg_tokens_per_sec) * 1000
            
            scored_models.append((model_id, score, error_rate, estimated_latency))
            
        # Sort by score (lower is better)
        scored_models.sort(key=lambda x: x[1])
        
        # Select best model
        best_model_id, best_score, best_error, best_latency = scored_models[0]
        best_model = self.models[best_model_id]
        
        # Prepare alternatives
        alternatives = []
        if return_alternatives and len(scored_models) > 1:
            alternatives = [(mid, score) for mid, score, _, _ in scored_models[1:4]]
            
        routing_latency = (time.time() - start_time) * 1000
        
        logger.debug(
            f"Routed to {best_model_id} (cluster={cluster_id}, "
            f"score={best_score:.3f}, routing={routing_latency:.2f}ms)"
        )
        
        return RoutingResultV2(
            model_id=best_model_id,
            model_path=best_model.model_path,
            size_mb=best_model.size_mb,
            tokens_per_sec=best_model.avg_tokens_per_sec,
            score=best_score,
            cluster_id=cluster_id,
            distance=float(distance),
            error_rate=best_error,
            estimated_latency_ms=best_latency,
            routing_latency_ms=routing_latency,
            alternatives=alternatives
        )
    
    def _filter_models(
        self,
        available_models: Optional[List[str]],
        required_capabilities: Optional[List[str]]
    ) -> List[str]:
        """Filter models based on availability and capabilities."""
        candidates = list(self.models.keys())
        
        # Filter by availability
        if available_models is not None:
            candidates = [m for m in candidates if m in available_models]
            
        # Filter by capabilities
        if required_capabilities:
            candidates = [
                m for m in candidates
                if all(cap in self.models[m].capabilities for cap in required_capabilities)
            ]
            
        return candidates
    
    def _calculate_lambda(self, cost_preference: float) -> float:
        """Calculate lambda from cost preference.
        
        High quality preference (1.0) = low lambda (less penalty for size)
        Low quality preference (0.0) = high lambda (more penalty for size)
        """
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
            'n_clusters': self.cluster_engine.n_clusters,
            'n_models': len(self.models),
            'embedding_model': self.cluster_engine.embedding_model,
            'embedding_dims': self.cluster_engine.embedding_dim,
            'lambda_range': (self.lambda_min, self.lambda_max),
            'default_cost_preference': self.default_cost_preference,
            'clustering_algorithm': self.metadata.get('clustering_algorithm', 'HDBSCAN'),
            'silhouette_score': self.metadata.get('silhouette_score'),
            'models': list(self.models.keys())
        }
    
    def get_cluster_stats(self) -> dict:
        """Get statistics about clusters."""
        return {
            'n_clusters': self.cluster_engine.n_clusters,
            'embedding_dim': self.cluster_engine.embedding_dim,
            'silhouette_score': self.metadata.get('silhouette_score'),
            'clustering_params': self.metadata.get('clustering_params')
        }
