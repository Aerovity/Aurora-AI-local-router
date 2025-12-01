"""Lightweight clustering engine optimized for mobile devices.

This module provides a stripped-down version of the ClusterEngine
that removes cloud dependencies and optimizes for on-device inference.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class MobileClusterEngine:
    """Lightweight cluster engine for mobile routing.

    Optimized for:
    - Low memory footprint (float16 storage)
    - Fast inference (<10ms)
    - No cloud dependencies
    - Minimal model size
    """

    def __init__(self):
        """Initialize empty mobile cluster engine."""
        self.n_clusters: Optional[int] = None
        self.cluster_centers: Optional[npt.NDArray[np.float32]] = None
        self.embedding_model_name: Optional[str] = None
        self.normalization_strategy: str = "l2"
        self.is_fitted: bool = False

    @classmethod
    def from_cluster_centers(
        cls,
        cluster_centers: npt.NDArray,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        normalization_strategy: str = "l2"
    ) -> "MobileClusterEngine":
        """Create engine from pre-computed cluster centers.

        Args:
            cluster_centers: Pre-computed K-means centers (n_clusters × feature_dim)
            embedding_model_name: Name of embedding model used
            normalization_strategy: Normalization method ('l2', 'l1', 'max')

        Returns:
            Configured MobileClusterEngine ready for prediction
        """
        engine = cls()
        engine.cluster_centers = cluster_centers.astype(np.float32)
        engine.n_clusters = cluster_centers.shape[0]
        engine.embedding_model_name = embedding_model_name
        engine.normalization_strategy = normalization_strategy
        engine.is_fitted = True

        logger.info(
            f"Loaded mobile cluster engine: {engine.n_clusters} clusters, "
            f"{cluster_centers.shape[1]} features"
        )

        return engine

    def assign_cluster(
        self,
        embedding: npt.NDArray[np.float32]
    ) -> Tuple[int, float]:
        """Assign embedding to nearest cluster.

        Args:
            embedding: Input embedding vector (already normalized)

        Returns:
            Tuple of (cluster_id, distance_to_centroid)

        Raises:
            ValueError: If engine not fitted
        """
        if not self.is_fitted:
            raise ValueError("Engine not fitted. Load cluster centers first.")

        # Ensure proper shape
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Normalize embedding
        embedding_normalized = normalize(
            embedding,
            norm=self.normalization_strategy
        ).astype(np.float32)

        # Compute distances to all cluster centers (cosine distance)
        # For normalized vectors: distance = 1 - cosine_similarity = 1 - dot_product
        similarities = np.dot(self.cluster_centers, embedding_normalized.T).squeeze()
        distances = 1.0 - similarities

        # Find nearest cluster
        cluster_id = int(np.argmin(distances))
        distance = float(distances[cluster_id])

        return cluster_id, distance

    def save(self, path: Path) -> None:
        """Save cluster engine to disk.

        Args:
            path: Output file path (will use .pkl extension)
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted engine")

        data = {
            'cluster_centers': self.cluster_centers,
            'n_clusters': self.n_clusters,
            'embedding_model_name': self.embedding_model_name,
            'normalization_strategy': self.normalization_strategy,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Saved mobile cluster engine to {path}")

    @classmethod
    def load(cls, path: Path) -> "MobileClusterEngine":
        """Load cluster engine from disk.

        Args:
            path: Path to saved engine file

        Returns:
            Loaded MobileClusterEngine
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        engine = cls.from_cluster_centers(
            cluster_centers=data['cluster_centers'],
            embedding_model_name=data['embedding_model_name'],
            normalization_strategy=data.get('normalization_strategy', 'l2')
        )

        logger.info(f"Loaded mobile cluster engine from {path}")
        return engine

    def get_memory_size(self) -> int:
        """Get approximate memory size in bytes.

        Returns:
            Memory size in bytes
        """
        if not self.is_fitted:
            return 0

        # Cluster centers size
        centers_size = self.cluster_centers.nbytes

        # Overhead for metadata (rough estimate)
        overhead = 1024

        return centers_size + overhead

    def __repr__(self) -> str:
        """String representation."""
        if not self.is_fitted:
            return "MobileClusterEngine(fitted=False)"

        memory_mb = self.get_memory_size() / (1024 * 1024)
        return (
            f"MobileClusterEngine("
            f"n_clusters={self.n_clusters}, "
            f"feature_dim={self.cluster_centers.shape[1]}, "
            f"memory={memory_mb:.2f}MB)"
        )
