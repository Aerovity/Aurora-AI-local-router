"""
Helper Utilities for AuroraAI Router

Common functions used across the package.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union, Optional


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file with error handling.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Parsed JSON as dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> None:
    """
    Save dictionary as JSON file.
    
    Args:
        data: Dictionary to save
        path: Output file path
        indent: JSON indentation (default: 2)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, default=_json_serializer)


def _json_serializer(obj):
    """Custom JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    L2 normalize an embedding vector.
    
    Args:
        embedding: Input embedding (1D or 2D array)
        
    Returns:
        Normalized embedding with unit norm
    """
    if embedding.ndim == 1:
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    else:
        # Batch normalization
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)  # Avoid division by zero
        return embedding / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score in [-1, 1]
    """
    a_norm = normalize_embedding(a)
    b_norm = normalize_embedding(b)
    return float(np.dot(a_norm, b_norm))


def cosine_similarity_batch(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and multiple candidates.
    
    Args:
        query: Query vector (1D)
        candidates: Candidate vectors (2D, shape [n, dim])
        
    Returns:
        Array of similarity scores
    """
    query_norm = normalize_embedding(query.reshape(1, -1))
    candidates_norm = normalize_embedding(candidates)
    return np.dot(candidates_norm, query_norm.T).flatten()


def compute_error_rate(
    predictions: List[str],
    ground_truth: List[str]
) -> float:
    """
    Compute error rate between predictions and ground truth.
    
    Args:
        predictions: List of predicted values
        ground_truth: List of correct values
        
    Returns:
        Error rate in [0, 1]
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    if len(predictions) == 0:
        return 0.0
    
    errors = sum(1 for p, g in zip(predictions, ground_truth) if p != g)
    return errors / len(predictions)


def find_best_model(
    error_rates: Dict[str, float],
    model_sizes: Optional[Dict[str, float]] = None,
    size_weight: float = 0.1
) -> str:
    """
    Find best model balancing error rate and size.
    
    Args:
        error_rates: Dictionary of model_id -> error_rate
        model_sizes: Optional dictionary of model_id -> size_mb
        size_weight: Weight for size penalty (default: 0.1)
        
    Returns:
        Best model ID
    """
    if not error_rates:
        raise ValueError("error_rates cannot be empty")
    
    if model_sizes is None or size_weight == 0:
        # Pure accuracy selection
        return min(error_rates.keys(), key=lambda m: error_rates[m])
    
    # Normalize sizes to [0, 1]
    max_size = max(model_sizes.values())
    
    scores = {}
    for model_id, error_rate in error_rates.items():
        size = model_sizes.get(model_id, max_size)
        size_penalty = (size / max_size) * size_weight
        scores[model_id] = error_rate + size_penalty
    
    return min(scores.keys(), key=lambda m: scores[m])


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    Split list into chunks of specified size.
    
    Args:
        lst: Input list
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_list(nested: List[List]) -> List:
    """Flatten a nested list."""
    return [item for sublist in nested for item in sublist]


def safe_mean(values: List[float], default: float = 0.0) -> float:
    """Compute mean with fallback for empty lists."""
    if not values:
        return default
    return sum(values) / len(values)


def safe_std(values: List[float], default: float = 0.0) -> float:
    """Compute standard deviation with fallback for empty lists."""
    if len(values) < 2:
        return default
    mean = safe_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def percentile(values: List[float], p: float) -> float:
    """
    Compute percentile of values.
    
    Args:
        values: List of values
        p: Percentile in [0, 100]
        
    Returns:
        Percentile value
    """
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    idx = (len(sorted_values) - 1) * p / 100
    lower = int(idx)
    upper = lower + 1
    
    if upper >= len(sorted_values):
        return sorted_values[-1]
    
    weight = idx - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
