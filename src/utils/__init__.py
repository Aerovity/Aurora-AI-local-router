"""
AuroraAI Router - Utilities Module

Common utility functions and helpers used across the router package.
"""

from .helpers import (
    load_json,
    save_json,
    normalize_embedding,
    cosine_similarity,
    compute_error_rate
)
from .logging import setup_logger, get_logger

__all__ = [
    'load_json',
    'save_json',
    'normalize_embedding',
    'cosine_similarity',
    'compute_error_rate',
    'setup_logger',
    'get_logger'
]
