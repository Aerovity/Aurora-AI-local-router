"""Mobile-optimized model router for on-device inference.

This router is designed to work with Cactus Compute for fully local
model selection without any API calls or cloud dependencies.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from .mobile_cluster_engine import MobileClusterEngine

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a model available for routing."""
    model_id: str
    model_path: str  # Path to model weights for Cactus
    size_mb: float
    avg_tokens_per_sec: float
    context_size: int = 2048
    capabilities: List[str] = None  # e.g., ['text', 'vision', 'tools']

    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = ['text']


@dataclass
class RoutingResult:
    """Result of model routing decision."""
    model_id: str
    model_path: str
    score: float
    cluster_id: int
    estimated_latency_ms: float
    alternatives: List[Tuple[str, float]] = None  # [(model_id, score), ...]

    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []


class MobileRouter:
    """Lightweight model router optimized for mobile devices.

    Features:
    - No cloud dependencies
    - Fast routing (<20ms)
    - Works with Cactus Compute
    - Minimal memory footprint
    """

    def __init__(
        self,
        cluster_engine: MobileClusterEngine,
        models: List[ModelInfo],
        error_rates: Dict[str, List[float]],
        lambda_min: float = 0.0,
        lambda_max: float = 2.0,
        default_cost_preference: float = 0.5
    ):
        """Initialize mobile router.

        Args:
            cluster_engine: Fitted mobile cluster engine
            models: List of available models
            error_rates: Dict mapping model_id to per-cluster error rates
            lambda_min: Minimum lambda for cost-quality tradeoff
            lambda_max: Maximum lambda for cost-quality tradeoff
            default_cost_preference: Default cost preference (0=fast, 1=quality)
        """
        self.cluster_engine = cluster_engine
        self.models = {m.model_id: m for m in models}
        self.error_rates = error_rates
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.default_cost_preference = default_cost_preference

        # Compute size normalization
        sizes = [m.size_mb for m in models]
        self.min_size = min(sizes)
        self.max_size = max(sizes)
        self.size_range = self.max_size - self.min_size

        logger.info(
            f"Initialized MobileRouter with {len(models)} models, "
            f"{cluster_engine.n_clusters} clusters"
        )

    def route(
        self,
        prompt_embedding: npt.NDArray[np.float32],
        available_models: Optional[List[str]] = None,
        cost_preference: Optional[float] = None,
        required_capabilities: Optional[List[str]] = None,
        return_alternatives: bool = False
    ) -> RoutingResult:
        """Route prompt to optimal model.

        Args:
            prompt_embedding: Pre-computed embedding of the prompt
            available_models: Optional list of model IDs to consider
            cost_preference: 0.0=fast/small, 1.0=quality/large (overrides default)
            required_capabilities: Required model capabilities (e.g., ['vision'])
            return_alternatives: Whether to return alternative model suggestions

        Returns:
            RoutingResult with selected model and metadata
        """
        start_time = time.time()

        # Determine which models to consider
        models_to_score = self._filter_models(available_models, required_capabilities)

        if not models_to_score:
            raise ValueError(
                f"No models available matching criteria: "
                f"available={available_models}, capabilities={required_capabilities}"
            )

        # Assign to cluster
        cluster_id, distance = self.cluster_engine.assign_cluster(prompt_embedding)

        # Calculate lambda parameter
        pref = cost_preference if cost_preference is not None else self.default_cost_preference
        lambda_param = self._calculate_lambda(pref)

        # Score all models
        scored_models = []
        for model_id in models_to_score:
            model_info = self.models[model_id]
            error_rate = self.error_rates[model_id][cluster_id]
            normalized_size = self._normalize_size(model_info.size_mb)

            # Score = error_rate + lambda * normalized_size
            score = error_rate + lambda_param * normalized_size

            # Estimate latency (rough approximation)
            # Assuming 100 tokens output, latency = 100 / tokens_per_sec * 1000
            estimated_latency = (100.0 / model_info.avg_tokens_per_sec) * 1000

            scored_models.append((model_id, score, estimated_latency))

        # Sort by score (lower is better)
        scored_models.sort(key=lambda x: x[1])

        # Select best model
        best_model_id, best_score, best_latency = scored_models[0]
        best_model_info = self.models[best_model_id]

        # Prepare alternatives
        alternatives = []
        if return_alternatives and len(scored_models) > 1:
            alternatives = [(mid, score) for mid, score, _ in scored_models[1:4]]

        routing_time = (time.time() - start_time) * 1000

        logger.info(
            f"Routed to {best_model_id} (cluster={cluster_id}, "
            f"score={best_score:.3f}, routing_time={routing_time:.2f}ms)"
        )

        return RoutingResult(
            model_id=best_model_id,
            model_path=best_model_info.model_path,
            score=best_score,
            cluster_id=cluster_id,
            estimated_latency_ms=best_latency,
            alternatives=alternatives
        )

    def route_from_text(
        self,
        prompt: str,
        embedding_function,  # Function that takes text and returns embedding
        available_models: Optional[List[str]] = None,
        cost_preference: Optional[float] = None,
        required_capabilities: Optional[List[str]] = None,
        return_alternatives: bool = False
    ) -> RoutingResult:
        """Route prompt text to optimal model.

        Convenience method that handles embedding extraction.

        Args:
            prompt: Input text prompt
            embedding_function: Function to extract embeddings from text
            available_models: Optional list of model IDs to consider
            cost_preference: 0.0=fast/small, 1.0=quality/large
            required_capabilities: Required model capabilities
            return_alternatives: Whether to return alternative suggestions

        Returns:
            RoutingResult with selected model
        """
        # Extract embedding
        embedding = embedding_function(prompt)

        # Ensure proper shape and type
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        embedding = embedding.astype(np.float32)

        return self.route(
            prompt_embedding=embedding,
            available_models=available_models,
            cost_preference=cost_preference,
            required_capabilities=required_capabilities,
            return_alternatives=return_alternatives
        )

    def _filter_models(
        self,
        available_models: Optional[List[str]],
        required_capabilities: Optional[List[str]]
    ) -> List[str]:
        """Filter models based on availability and capabilities.

        Args:
            available_models: Optional list of model IDs to restrict to
            required_capabilities: Required capabilities

        Returns:
            List of model IDs to score
        """
        # Start with all models
        candidates = list(self.models.keys())

        # Filter by availability
        if available_models is not None:
            candidates = [mid for mid in candidates if mid in available_models]

        # Filter by capabilities
        if required_capabilities:
            candidates = [
                mid for mid in candidates
                if all(cap in self.models[mid].capabilities for cap in required_capabilities)
            ]

        return candidates

    def _calculate_lambda(self, cost_preference: float) -> float:
        """Calculate lambda parameter from cost preference.

        Args:
            cost_preference: 0.0=fast/cheap, 1.0=quality

        Returns:
            Lambda value (higher = more penalty for size/cost)
        """
        # Invert: high quality preference = low lambda
        lambda_param = self.lambda_max - cost_preference * (
            self.lambda_max - self.lambda_min
        )
        return lambda_param

    def _normalize_size(self, size_mb: float) -> float:
        """Normalize model size to [0, 1] range.

        Args:
            size_mb: Model size in MB

        Returns:
            Normalized size
        """
        if self.size_range < 1e-6:
            return 0.0

        return (size_mb - self.min_size) / self.size_range

    @classmethod
    def from_profile(
        cls,
        profile_path: Path,
        models: List[ModelInfo]
    ) -> "MobileRouter":
        """Load router from profile file.

        Args:
            profile_path: Path to saved profile (contains cluster centers and error rates)
            models: List of model information

        Returns:
            Configured MobileRouter
        """
        import json

        profile_path = Path(profile_path)

        with open(profile_path, 'r') as f:
            profile_data = json.load(f)

        # Load cluster engine
        cluster_centers = np.array(
            profile_data['cluster_centers']['cluster_centers'],
            dtype=np.float32
        )

        cluster_engine = MobileClusterEngine.from_cluster_centers(
            cluster_centers=cluster_centers,
            embedding_model_name=profile_data['metadata']['embedding_model']
        )

        # Extract error rates
        error_rates = profile_data['llm_profiles']

        # Extract metadata
        metadata = profile_data['metadata']

        return cls(
            cluster_engine=cluster_engine,
            models=models,
            error_rates=error_rates,
            lambda_min=metadata.get('lambda_min', 0.0),
            lambda_max=metadata.get('lambda_max', 2.0),
            default_cost_preference=metadata.get('default_cost_preference', 0.5)
        )

    def get_supported_models(self) -> List[str]:
        """Get list of supported model IDs.

        Returns:
            List of model IDs
        """
        return list(self.models.keys())

    def get_cluster_info(self) -> Dict:
        """Get information about loaded clusters.

        Returns:
            Dictionary with cluster statistics
        """
        return {
            'n_clusters': self.cluster_engine.n_clusters,
            'embedding_model': self.cluster_engine.embedding_model_name,
            'supported_models': self.get_supported_models(),
            'lambda_min': self.lambda_min,
            'lambda_max': self.lambda_max,
            'default_cost_preference': self.default_cost_preference,
        }
