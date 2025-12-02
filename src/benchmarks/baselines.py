"""
Baseline Routers for Benchmarking.

Simple routing strategies to compare against learned routers.
"""

import random
from typing import Optional, List
from dataclasses import dataclass

from ..config import CACTUS_MODELS, MODEL_BY_ID


@dataclass
class BaselineResult:
    """Result from a baseline router."""
    model_id: str
    size_mb: float
    tokens_per_sec: float
    latency_ms: float = 0.01


class BaselineRouters:
    """Collection of baseline routing strategies."""
    
    @staticmethod
    def always_largest(prompt: str, cost_preference: float = 0.5) -> BaselineResult:
        """Always select the largest model (quality baseline)."""
        m = CACTUS_MODELS[-1]  # lfm2-vl-1.6b
        return BaselineResult(
            model_id=m.model_id,
            size_mb=m.size_mb,
            tokens_per_sec=m.avg_tokens_per_sec
        )
    
    @staticmethod
    def always_smallest(prompt: str, cost_preference: float = 0.5) -> BaselineResult:
        """Always select the smallest model (speed baseline)."""
        m = CACTUS_MODELS[0]  # gemma-270m
        return BaselineResult(
            model_id=m.model_id,
            size_mb=m.size_mb,
            tokens_per_sec=m.avg_tokens_per_sec
        )
    
    @staticmethod
    def random_selection(prompt: str, cost_preference: float = 0.5) -> BaselineResult:
        """Random model selection."""
        m = random.choice(CACTUS_MODELS)
        return BaselineResult(
            model_id=m.model_id,
            size_mb=m.size_mb,
            tokens_per_sec=m.avg_tokens_per_sec
        )
    
    @staticmethod
    def size_weighted_random(prompt: str, cost_preference: float = 0.5) -> BaselineResult:
        """Random selection weighted by inverse size (smaller = more likely)."""
        sizes = [m.size_mb for m in CACTUS_MODELS]
        max_size = max(sizes)
        weights = [(max_size - s + 100) for s in sizes]
        
        m = random.choices(CACTUS_MODELS, weights=weights, k=1)[0]
        return BaselineResult(
            model_id=m.model_id,
            size_mb=m.size_mb,
            tokens_per_sec=m.avg_tokens_per_sec
        )
    
    @staticmethod
    def cost_preference_based(prompt: str, cost_preference: float = 0.5) -> BaselineResult:
        """Select model based on cost preference only (no clustering)."""
        # Map cost_preference to model index
        # 0.0 -> smallest, 1.0 -> largest
        idx = int(cost_preference * (len(CACTUS_MODELS) - 1))
        idx = max(0, min(len(CACTUS_MODELS) - 1, idx))
        
        m = CACTUS_MODELS[idx]
        return BaselineResult(
            model_id=m.model_id,
            size_mb=m.size_mb,
            tokens_per_sec=m.avg_tokens_per_sec
        )
    
    @staticmethod
    def medium_model(prompt: str, cost_preference: float = 0.5) -> BaselineResult:
        """Always select a medium-sized model (balanced baseline)."""
        # Select gemma-1b (middle of the pack)
        m = MODEL_BY_ID.get('gemma-1b', CACTUS_MODELS[len(CACTUS_MODELS)//2])
        return BaselineResult(
            model_id=m.model_id,
            size_mb=m.size_mb,
            tokens_per_sec=m.avg_tokens_per_sec
        )
