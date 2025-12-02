"""
Utility functions for training and profile generation.
"""

import numpy as np
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import time


def simulate_model_error_rate(model_size_mb: int, cluster_samples: int, seed: int = 42) -> float:
    """
    Simulate error rate for a model on a cluster.

    This is a placeholder that generates synthetic error rates based on model size.

    TODO: Replace with actual Cactus inference when running on Mac:
        for sample in cluster_samples:
            response = cactus_complete(model, sample['question'])
            is_correct = evaluate_answer(response, sample['answer'])
            error_rate = 1.0 - accuracy

    Args:
        model_size_mb: Model size in megabytes (larger = better assumed)
        cluster_samples: Number of samples in cluster
        seed: Random seed for reproducibility

    Returns:
        Simulated error rate (0.0 to 1.0)
    """
    # Simulate: larger models have lower error rates
    base_accuracy = min(0.95, 0.40 + (model_size_mb / 1500) * 0.55)

    # Add some randomness per model
    np.random.seed(seed)
    noise = np.random.uniform(-0.05, 0.05)
    accuracy = np.clip(base_accuracy + noise, 0.30, 0.95)

    error_rate = 1.0 - accuracy
    return float(error_rate)


def compute_cluster_error_rates(
    models: List[Dict],
    cluster_labels: np.ndarray,
    samples: List[Dict]
) -> Dict[str, List[float]]:
    """
    Compute error rates for each model on each cluster.

    Args:
        models: List of model configurations
        cluster_labels: Cluster assignment for each sample
        samples: List of MMLU samples

    Returns:
        Dictionary mapping model_id to list of error rates per cluster
    """
    error_rates = {}
    unique_clusters = sorted(set(cluster_labels) - {-1})  # Exclude noise

    print(f"\n🔬 Computing error rates for {len(models)} models across {len(unique_clusters)} clusters...")

    for model in models:
        model_id = model['model_id']
        model_size = model['size_mb']
        rates = []

        for cluster_id in unique_clusters:
            # Get samples in this cluster
            cluster_mask = (cluster_labels == cluster_id)
            n_samples = cluster_mask.sum()

            if n_samples == 0:
                rates.append(0.5)  # Default for empty clusters
                continue

            # Simulate error rate (replace with real inference on Mac)
            seed = hash(f"{model_id}_{cluster_id}") % (2**31)
            error_rate = simulate_model_error_rate(model_size, n_samples, seed)
            rates.append(error_rate)

        error_rates[model_id] = rates
        avg_error = np.mean(rates)
        print(f"  {model_id:<20}: {avg_error:5.2%} avg error")

    return error_rates


def save_profile(
    profile: Dict,
    output_path: str,
    indent: int = 2
):
    """
    Save router profile to JSON file.

    Args:
        profile: Profile dictionary
        output_path: Output file path
        indent: JSON indentation (default: 2)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    profile_copy = {}
    for key, value in profile.items():
        if isinstance(value, np.ndarray):
            profile_copy[key] = value.tolist()
        elif isinstance(value, dict):
            profile_copy[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    profile_copy[key][k] = v.tolist()
                else:
                    profile_copy[key][k] = v
        else:
            profile_copy[key] = value

    with open(output_path, 'w') as f:
        json.dump(profile_copy, f, indent=indent)

    # Get file size
    file_size_kb = output_path.stat().st_size / 1024
    print(f"\n✅ Profile saved to: {output_path}")
    print(f"   File size: {file_size_kb:.1f} KB")


def load_profile(profile_path: str) -> Dict:
    """
    Load router profile from JSON file.

    Args:
        profile_path: Path to profile JSON file

    Returns:
        Profile dictionary with numpy arrays
    """
    with open(profile_path, 'r') as f:
        profile = json.load(f)

    # Convert lists back to numpy arrays
    if 'cluster_centers' in profile and 'cluster_centers' in profile['cluster_centers']:
        profile['cluster_centers']['cluster_centers'] = np.array(
            profile['cluster_centers']['cluster_centers']
        )

    return profile


def print_cluster_distribution(labels: np.ndarray, topics: Optional[List[str]] = None):
    """
    Print cluster size distribution.

    Args:
        labels: Cluster labels
        topics: Optional list of topic names for analysis
    """
    unique_labels = sorted(set(labels))

    print("\n📊 Cluster Distribution:")
    print("=" * 60)

    for label in unique_labels:
        count = (labels == label).sum()
        percentage = 100 * count / len(labels)

        if label == -1:
            print(f"  Noise:      {count:4d} samples ({percentage:5.1f}%)")
        else:
            print(f"  Cluster {label:2d}: {count:4d} samples ({percentage:5.1f}%)")

    print("=" * 60)


def save_embeddings_cache(
    embeddings: np.ndarray,
    cache_path: str,
    metadata: Optional[Dict] = None
):
    """
    Save embeddings to disk for faster re-runs.

    Args:
        embeddings: Embeddings array
        cache_path: Path to save cache file
        metadata: Optional metadata to save alongside embeddings
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as compressed numpy file
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        metadata=json.dumps(metadata) if metadata else ""
    )

    file_size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"💾 Embeddings cached to: {cache_path} ({file_size_mb:.1f} MB)")


def load_embeddings_cache(cache_path: str) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Load embeddings from cache.

    Args:
        cache_path: Path to cache file

    Returns:
        Tuple of (embeddings, metadata)
    """
    cache_path = Path(cache_path)

    if not cache_path.exists():
        return None, None

    data = np.load(cache_path, allow_pickle=True)
    embeddings = data['embeddings']

    metadata = None
    if 'metadata' in data and data['metadata']:
        metadata = json.loads(str(data['metadata']))

    print(f"💾 Loaded embeddings from cache: {cache_path}")
    print(f"   Shape: {embeddings.shape}")

    return embeddings, metadata


class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self, name: str = "Operation", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        if self.verbose:
            print(f"⏱️  {self.name}...")
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time

        if self.verbose:
            if elapsed < 60:
                print(f"✅ {self.name} completed in {elapsed:.2f}s")
            else:
                minutes = int(elapsed // 60)
                seconds = elapsed % 60
                print(f"✅ {self.name} completed in {minutes}m {seconds:.1f}s")

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
