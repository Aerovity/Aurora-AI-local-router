"""
Cluster Engine for V2 CACTUS Router.

Handles cluster assignment and distance calculations.
Optimized for fast inference on embedded vectors.
"""

import numpy as np
from typing import Tuple, Optional


class ClusterEngine:
    """Cluster assignment engine for CACTUS router.
    
    Handles:
    - Loading cluster centers from profile
    - Fast nearest-cluster lookup
    - Distance calculations
    """
    
    def __init__(
        self,
        cluster_centers: np.ndarray,
        embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    ):
        """Initialize cluster engine.
        
        Args:
            cluster_centers: Array of shape (n_clusters, embedding_dim)
            embedding_model: Name of embedding model used to train clusters
        """
        self.cluster_centers = np.asarray(cluster_centers, dtype=np.float32)
        self.n_clusters = len(cluster_centers)
        self.embedding_dim = cluster_centers.shape[1]
        self.embedding_model = embedding_model
        
        # Pre-compute norms for faster distance calculation
        self._center_norms = np.linalg.norm(cluster_centers, axis=1)
        
    def assign_cluster(self, embedding: np.ndarray) -> Tuple[int, float]:
        """Assign embedding to nearest cluster.
        
        Args:
            embedding: Embedding vector of shape (embedding_dim,)
            
        Returns:
            Tuple of (cluster_id, distance)
        """
        embedding = np.asarray(embedding, dtype=np.float32)
        
        # Handle dimension mismatch
        if len(embedding) != self.embedding_dim:
            if len(embedding) > self.embedding_dim:
                # Truncate (Matryoshka-style)
                embedding = embedding[:self.embedding_dim]
            else:
                raise ValueError(
                    f"Embedding dimension {len(embedding)} doesn't match "
                    f"cluster dimension {self.embedding_dim}"
                )
        
        # Calculate Euclidean distances
        distances = np.linalg.norm(self.cluster_centers - embedding, axis=1)
        
        cluster_id = int(np.argmin(distances))
        distance = float(distances[cluster_id])
        
        return cluster_id, distance
    
    def assign_batch(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Assign batch of embeddings to nearest clusters.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            
        Returns:
            Tuple of (cluster_ids, distances) arrays
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        
        # Handle dimension mismatch
        if embeddings.shape[1] != self.embedding_dim:
            if embeddings.shape[1] > self.embedding_dim:
                embeddings = embeddings[:, :self.embedding_dim]
            else:
                raise ValueError(
                    f"Embedding dimension {embeddings.shape[1]} doesn't match "
                    f"cluster dimension {self.embedding_dim}"
                )
        
        # Calculate all pairwise distances
        # distances[i, j] = distance from embedding i to cluster j
        distances = np.zeros((len(embeddings), self.n_clusters), dtype=np.float32)
        
        for j in range(self.n_clusters):
            diff = embeddings - self.cluster_centers[j]
            distances[:, j] = np.linalg.norm(diff, axis=1)
            
        cluster_ids = np.argmin(distances, axis=1)
        min_distances = distances[np.arange(len(embeddings)), cluster_ids]
        
        return cluster_ids, min_distances
    
    def get_cluster_probabilities(self, embedding: np.ndarray) -> np.ndarray:
        """Get soft cluster assignment probabilities.
        
        Uses inverse distance weighting for soft assignment.
        
        Args:
            embedding: Embedding vector
            
        Returns:
            Array of cluster probabilities (sums to 1)
        """
        embedding = np.asarray(embedding, dtype=np.float32)
        
        if len(embedding) > self.embedding_dim:
            embedding = embedding[:self.embedding_dim]
            
        distances = np.linalg.norm(self.cluster_centers - embedding, axis=1)
        
        # Inverse distance weighting (add small epsilon to avoid div by zero)
        inv_distances = 1.0 / (distances + 1e-8)
        probabilities = inv_distances / inv_distances.sum()
        
        return probabilities
    
    @classmethod
    def from_profile(cls, profile_data: dict) -> "ClusterEngine":
        """Create cluster engine from profile data.
        
        Args:
            profile_data: Loaded profile dictionary
            
        Returns:
            Configured ClusterEngine
        """
        cluster_centers = np.array(
            profile_data['cluster_centers']['cluster_centers'],
            dtype=np.float32
        )
        
        embedding_model = profile_data.get('metadata', {}).get(
            'embedding_model', 
            'nomic-ai/nomic-embed-text-v1.5'
        )
        
        return cls(
            cluster_centers=cluster_centers,
            embedding_model=embedding_model
        )
    
    def get_info(self) -> dict:
        """Get cluster engine information."""
        return {
            'n_clusters': self.n_clusters,
            'embedding_dim': self.embedding_dim,
            'embedding_model': self.embedding_model
        }
