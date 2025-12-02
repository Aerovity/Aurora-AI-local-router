"""
Shared configuration for AuroraAI Router.

Contains model definitions, paths, and constants used by both v1 and v2 routers.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a single model."""
    model_id: str
    model_path: str
    size_mb: float
    avg_tokens_per_sec: float
    capabilities: List[str] = field(default_factory=lambda: ['text'])
    context_size: int = 2048
    
    def to_dict(self) -> dict:
        return {
            'model_id': self.model_id,
            'model_path': self.model_path,
            'size_mb': self.size_mb,
            'avg_tokens_per_sec': self.avg_tokens_per_sec,
            'capabilities': self.capabilities,
            'context_size': self.context_size,
        }


# Cactus-compatible Small Language Models
CACTUS_MODELS = [
    ModelConfig(
        model_id='gemma-270m',
        model_path='google/gemma-3-270m-it',
        size_mb=172,
        avg_tokens_per_sec=173,
        capabilities=['text']
    ),
    ModelConfig(
        model_id='lfm2-350m',
        model_path='LiquidAI/LFM2-350M',
        size_mb=233,
        avg_tokens_per_sec=145,
        capabilities=['text', 'tools', 'embed']
    ),
    ModelConfig(
        model_id='smollm-360m',
        model_path='HuggingFaceTB/SmolLM2-360m-Instruct',
        size_mb=227,
        avg_tokens_per_sec=150,
        capabilities=['text', 'embed']
    ),
    ModelConfig(
        model_id='qwen-600m',
        model_path='Qwen/Qwen3-0.6B',
        size_mb=411,
        avg_tokens_per_sec=120,
        capabilities=['text', 'tools', 'embed']
    ),
    ModelConfig(
        model_id='lfm2-vl-450m',
        model_path='LiquidAI/LFM2-VL-450M',
        size_mb=306,
        avg_tokens_per_sec=130,
        capabilities=['text', 'vision', 'embed']
    ),
    ModelConfig(
        model_id='lfm2-700m',
        model_path='LiquidAI/LFM2-700M',
        size_mb=486,
        avg_tokens_per_sec=110,
        capabilities=['text', 'tools', 'embed']
    ),
    ModelConfig(
        model_id='gemma-1b',
        model_path='google/gemma-3-1b-it',
        size_mb=642,
        avg_tokens_per_sec=100,
        capabilities=['text']
    ),
    ModelConfig(
        model_id='lfm2-1.2b',
        model_path='LiquidAI/LFM2-1.2B',
        size_mb=722,
        avg_tokens_per_sec=95,
        capabilities=['text', 'tools', 'embed']
    ),
    ModelConfig(
        model_id='lfm2-1.2b-tools',
        model_path='LiquidAI/LFM2-1.2B-Tools',
        size_mb=722,
        avg_tokens_per_sec=95,
        capabilities=['text', 'tools', 'embed']
    ),
    ModelConfig(
        model_id='qwen-1.7b',
        model_path='Qwen/Qwen3-1.7B',
        size_mb=1161,
        avg_tokens_per_sec=75,
        capabilities=['text', 'tools', 'embed']
    ),
    ModelConfig(
        model_id='smollm-1.7b',
        model_path='HuggingFaceTB/SmolLM2-1.7B-Instruct',
        size_mb=1161,
        avg_tokens_per_sec=72,
        capabilities=['text', 'embed']
    ),
    ModelConfig(
        model_id='lfm2-vl-1.6b',
        model_path='LiquidAI/LFM2-VL-1.6B',
        size_mb=1440,
        avg_tokens_per_sec=60,
        capabilities=['text', 'vision', 'embed']
    ),
]

# Quick lookup by model_id
MODEL_BY_ID = {m.model_id: m for m in CACTUS_MODELS}
MODEL_IDS = [m.model_id for m in CACTUS_MODELS]

# As list of dicts for backwards compatibility
CACTUS_MODELS_DICT = [m.to_dict() for m in CACTUS_MODELS]


# =============================================================================
# EMBEDDING MODEL CONFIGURATIONS
# =============================================================================

@dataclass
class EmbeddingConfig:
    """Configuration for an embedding model."""
    name: str
    model_id: str
    dimensions: int
    size_mb: float
    mobile_friendly: bool
    requires_trust_remote: bool = False
    
    
# V1: Sentence Transformers (lighter, faster)
EMBEDDING_V1 = EmbeddingConfig(
    name="MiniLM-L6",
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    dimensions=384,
    size_mb=80,
    mobile_friendly=True
)

# V2: Nomic Embed (better quality, larger)
EMBEDDING_V2 = EmbeddingConfig(
    name="Nomic-Embed-v1.5",
    model_id="nomic-ai/nomic-embed-text-v1.5",
    dimensions=768,
    size_mb=500,
    mobile_friendly=False,
    requires_trust_remote=True
)

# Mobile-optimized alternatives
EMBEDDING_MOBILE = [
    EmbeddingConfig(
        name="BGE-Micro",
        model_id="BAAI/bge-micro-v2",
        dimensions=384,
        size_mb=15,
        mobile_friendly=True
    ),
    EmbeddingConfig(
        name="GTE-Small",
        model_id="thenlper/gte-small",
        dimensions=384,
        size_mb=60,
        mobile_friendly=True
    ),
]


# =============================================================================
# PATH CONFIGURATIONS  
# =============================================================================

class PathConfig:
    """Standard paths for the project."""
    
    ROOT = Path(__file__).parent.parent
    
    # Profiles
    PROFILES = ROOT / "profiles"
    PROFILES_V1 = PROFILES / "v1"
    PROFILES_V2 = PROFILES / "v2"
    PROFILES_PRODUCTION = PROFILES / "production"
    
    # Cactus integration (trained profiles)
    CACTUS_INTEGRATION = ROOT / "cactus-integration"
    CACTUS_PROFILES = CACTUS_INTEGRATION / "profiles"
    
    # Default profile path (trained CACTUS profile)
    PROFILE_PATH = CACTUS_PROFILES / "cactus_profile.json"
    
    # Training outputs
    TRAINING = ROOT / "training"
    
    # SDKs
    SDKS = ROOT / "sdks"
    
    @classmethod
    def get_profile(cls, version: str = "v2", name: str = "default") -> Path:
        """Get path to a profile file."""
        if version == "v1":
            return cls.PROFILES_V1 / f"{name}_profile.json"
        elif version == "v2":
            return cls.PROFILES_V2 / f"{name}_profile.json"
        elif version == "cactus":
            return cls.CACTUS_PROFILES / f"{name}_profile.json"
        else:
            return cls.PROFILES_PRODUCTION / f"{name}_profile.json"
    
    @classmethod
    def ensure_dirs(cls):
        """Create all required directories."""
        for dir_path in [cls.PROFILES, cls.PROFILES_V1, cls.PROFILES_V2, 
                         cls.PROFILES_PRODUCTION, cls.TRAINING]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Backwards compatibility alias
Paths = PathConfig


# =============================================================================
# ROUTING CONFIGURATIONS
# =============================================================================

@dataclass
class RoutingConfig:
    """Configuration for the routing algorithm."""
    lambda_min: float = 0.0
    lambda_max: float = 2.0
    default_cost_preference: float = 0.5
    
    # Cluster settings
    min_cluster_size: int = 20
    min_samples: int = 15
    
    # Performance thresholds
    max_routing_latency_ms: float = 50.0
    target_savings_percent: float = 30.0


DEFAULT_ROUTING_CONFIG = RoutingConfig()
