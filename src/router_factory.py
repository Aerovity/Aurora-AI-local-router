"""
Router Factory - Unified interface for creating routers.

Automatically detects profile version and creates appropriate router.
"""

import json
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

from .config import CACTUS_MODELS, ModelConfig


class RouterVersion(Enum):
    """Router version enumeration."""
    V1 = "1.0"  # MiniLM embeddings
    V2 = "2.0"  # Nomic/CACTUS embeddings
    AUTO = "auto"  # Auto-detect from profile


def detect_profile_version(profile_path: Path) -> RouterVersion:
    """Detect router version from profile file.
    
    Args:
        profile_path: Path to profile JSON
        
    Returns:
        RouterVersion enum
    """
    with open(profile_path, 'r') as f:
        profile = json.load(f)
        
    metadata = profile.get('metadata', {})
    
    # Check embedding model
    embedding_model = metadata.get('embedding_model', '')
    
    if 'nomic' in embedding_model.lower():
        return RouterVersion.V2
    elif 'minilm' in embedding_model.lower() or 'sentence-transformers' in embedding_model.lower():
        return RouterVersion.V1
        
    # Check feature dimensions
    feature_dim = metadata.get('feature_dim', 0)
    if feature_dim == 768:
        return RouterVersion.V2
    elif feature_dim == 384:
        return RouterVersion.V1
        
    # Check for CACTUS-specific fields
    if metadata.get('clustering_algorithm') == 'HDBSCAN':
        return RouterVersion.V2
        
    # Default to V1 for backwards compatibility
    return RouterVersion.V1


def create_router(
    profile_path: Union[str, Path],
    version: RouterVersion = RouterVersion.AUTO,
    models: Optional[List[ModelConfig]] = None,
    auto_load_embedder: bool = True
):
    """Create a router from profile file.
    
    Factory function that creates the appropriate router version
    based on the profile or explicit version parameter.
    
    Args:
        profile_path: Path to router profile JSON
        version: Router version (V1, V2, or AUTO for detection)
        models: Optional list of models (uses defaults if not provided)
        auto_load_embedder: Whether to auto-load embedding model
        
    Returns:
        RouterV1 or CactusRouter instance
        
    Example:
        >>> # Auto-detect version
        >>> router = create_router("profiles/v2/cactus_profile.json")
        >>> result = router.route("Hello world")
        
        >>> # Explicit V1
        >>> router = create_router("profiles/v1/profile.json", version=RouterVersion.V1)
    """
    profile_path = Path(profile_path)
    
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")
        
    # Detect version if auto
    if version == RouterVersion.AUTO:
        version = detect_profile_version(profile_path)
        
    # Use default models if not provided
    if models is None:
        models = CACTUS_MODELS
        
    # Create appropriate router
    if version == RouterVersion.V1:
        from .v1 import RouterV1
        return RouterV1.from_profile(profile_path, models)
    else:
        from .v2 import CactusRouter
        return CactusRouter.from_profile(
            profile_path, 
            models, 
            auto_load_embedder=auto_load_embedder
        )


class UnifiedRouter:
    """Unified router interface that wraps V1 or V2 router.
    
    Provides a consistent API regardless of underlying implementation.
    Useful for applications that need to support both versions.
    """
    
    def __init__(
        self,
        profile_path: Union[str, Path],
        version: RouterVersion = RouterVersion.AUTO,
        models: Optional[List[ModelConfig]] = None
    ):
        """Initialize unified router.
        
        Args:
            profile_path: Path to router profile
            version: Router version
            models: Optional model list
        """
        self.profile_path = Path(profile_path)
        self._router = create_router(profile_path, version, models)
        self.version = version if version != RouterVersion.AUTO else detect_profile_version(self.profile_path)
        
    def route(
        self,
        prompt: str,
        cost_preference: float = 0.5,
        available_models: Optional[List[str]] = None,
        return_alternatives: bool = False
    ) -> dict:
        """Route a prompt to optimal model.
        
        Args:
            prompt: Input text
            cost_preference: 0.0=fast, 1.0=quality
            available_models: Optional filter
            return_alternatives: Include alternatives
            
        Returns:
            Dict with routing result
        """
        result = self._router.route(
            prompt=prompt,
            cost_preference=cost_preference,
            available_models=available_models,
            return_alternatives=return_alternatives
        )
        
        return result.to_dict()
    
    def get_info(self) -> dict:
        """Get router information."""
        info = self._router.get_info()
        info['wrapper_version'] = 'unified'
        return info
    
    @property
    def is_v2(self) -> bool:
        """Check if using V2 (CACTUS) router."""
        return self.version == RouterVersion.V2
