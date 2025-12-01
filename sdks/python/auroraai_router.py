"""Python SDK for AuroraAI Mobile Router.

Simple API for integrating the router with Python applications.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core import MobileRouter, ModelInfo, RoutingResult
from core import ProfileConverter, MobileClusterEngine

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class AuroraAIRouter:
    """High-level API for mobile model routing."""

    def __init__(
        self,
        profile_path: str,
        models: list,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        auto_load_embeddings: bool = True
    ):
        """Initialize router.

        Args:
            profile_path: Path to router profile JSON
            models: List of ModelInfo objects or dicts
            embedding_model_name: Name of embedding model to use
            auto_load_embeddings: Whether to auto-load embedding model

        Example:
            >>> models = [
            ...     {'model_id': 'gemma-270m', 'model_path': 'weights/gemma-270m',
            ...      'size_mb': 172, 'avg_tokens_per_sec': 173}
            ... ]
            >>> router = AuroraAIRouter('profile.json', models)
            >>> result = router.route("Hello world")
        """
        # Convert dicts to ModelInfo if needed
        if models and isinstance(models[0], dict):
            models = [ModelInfo(**m) for m in models]

        self.router = MobileRouter.from_profile(Path(profile_path), models)
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None

        if auto_load_embeddings and HAS_SENTENCE_TRANSFORMERS:
            self.load_embedding_model()

    def load_embedding_model(self):
        """Load embedding model (if not already loaded)."""
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

    def route(
        self,
        prompt: str,
        available_models: list = None,
        cost_preference: float = 0.5,
        return_alternatives: bool = False
    ) -> RoutingResult:
        """Route a prompt to the optimal model.

        Args:
            prompt: Input text prompt
            available_models: Optional list of model IDs to consider
            cost_preference: 0.0 = fast/small, 1.0 = quality/large
            return_alternatives: Whether to return alternative suggestions

        Returns:
            RoutingResult with selected model and metadata

        Example:
            >>> result = router.route("Explain quantum physics", cost_preference=0.8)
            >>> print(result.model_id)  # 'qwen-1.7b'
        """
        if self.embedding_model is None:
            self.load_embedding_model()

        def get_embedding(text):
            return self.embedding_model.encode(text, normalize_embeddings=False)

        return self.router.route_from_text(
            prompt=prompt,
            embedding_function=get_embedding,
            available_models=available_models,
            cost_preference=cost_preference,
            return_alternatives=return_alternatives
        )

    def get_info(self) -> dict:
        """Get router information.

        Returns:
            Dictionary with router metadata
        """
        return self.router.get_cluster_info()


__all__ = ['AuroraAIRouter', 'ModelInfo', 'RoutingResult']
