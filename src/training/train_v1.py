"""
V1 Trainer - KMeans clustering with MiniLM embeddings.

Lightweight training for mobile deployment.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..v1.embedder import MiniLMEmbedder
from ..config import CACTUS_MODELS, CACTUS_MODELS_DICT


class TrainerV1:
    """V1 Profile Trainer using MiniLM and KMeans.
    
    Creates router profiles using:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dims)
    - KMeans clustering
    - Simulated error rates based on model size
    """
    
    VERSION = "1.0"
    
    def __init__(
        self,
        n_clusters: int = 5,
        embedder: Optional[MiniLMEmbedder] = None
    ):
        """Initialize trainer.
        
        Args:
            n_clusters: Number of clusters for KMeans
            embedder: Optional embedder instance
        """
        self.n_clusters = n_clusters
        self.embedder = embedder or MiniLMEmbedder()
        
    def train(
        self,
        texts: List[str],
        output_path: Path,
        models: Optional[List[dict]] = None
    ) -> dict:
        """Train a V1 profile.
        
        Args:
            texts: Training texts
            output_path: Path to save profile
            models: Optional model definitions
            
        Returns:
            Profile metadata
        """
        print(f"🚀 Training V1 profile with {len(texts)} samples...")
        
        # Generate embeddings
        print("  Generating MiniLM embeddings...")
        embeddings = self.embedder.embed_batch(texts)
        print(f"  ✅ Generated {len(embeddings)} embeddings ({embeddings.shape[1]} dims)")
        
        # Cluster
        print(f"  Running KMeans (K={self.n_clusters})...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score
        sil_score = silhouette_score(embeddings, labels)
        print(f"  ✅ Clustering complete (silhouette={sil_score:.4f})")
        
        # Get cluster centers
        cluster_centers = kmeans.cluster_centers_
        
        # Use provided models or defaults
        if models is None:
            models = CACTUS_MODELS_DICT
            
        # Compute error rates
        print("  Computing error rates...")
        error_rates = self._compute_error_rates(labels, models)
        
        # Build profile
        profile = {
            'version': self.VERSION,
            'metadata': {
                'n_clusters': self.n_clusters,
                'feature_dim': embeddings.shape[1],
                'embedding_model': self.embedder.model_name,
                'lambda_min': 0.0,
                'lambda_max': 2.0,
                'default_cost_preference': 0.5,
                'silhouette_score': float(sil_score),
                'clustering_algorithm': 'KMeans',
                'n_samples': len(texts),
                'created_at': datetime.now().isoformat()
            },
            'cluster_centers': {
                'n_clusters': self.n_clusters,
                'feature_dim': embeddings.shape[1],
                'cluster_centers': cluster_centers.tolist()
            },
            'llm_profiles': error_rates,
            'models': models
        }
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(profile, f, indent=2)
            
        size_kb = output_path.stat().st_size / 1024
        print(f"  ✅ Profile saved to {output_path} ({size_kb:.1f} KB)")
        
        return profile['metadata']
    
    def _compute_error_rates(
        self,
        labels: np.ndarray,
        models: List[dict]
    ) -> Dict[str, List[float]]:
        """Compute simulated error rates based on model size.
        
        Larger models have lower error rates.
        Error rates vary by cluster to simulate topic difficulty.
        """
        error_rates = {}
        
        # Get size range
        sizes = [m['size_mb'] for m in models]
        min_size, max_size = min(sizes), max(sizes)
        size_range = max_size - min_size
        
        # Base error rates by cluster (simulated difficulty)
        cluster_difficulty = np.random.uniform(0.3, 0.7, self.n_clusters)
        
        for model in models:
            model_id = model['model_id']
            size_mb = model['size_mb']
            
            # Normalized size (0=smallest, 1=largest)
            norm_size = (size_mb - min_size) / size_range if size_range > 0 else 0.5
            
            # Base error rate inversely proportional to size
            base_error = 0.6 - norm_size * 0.5  # Range: 0.1 to 0.6
            
            # Vary by cluster
            rates = []
            for c in range(self.n_clusters):
                cluster_factor = cluster_difficulty[c]
                error = base_error * cluster_factor
                error = max(0.01, min(0.99, error + np.random.uniform(-0.05, 0.05)))
                rates.append(float(error))
                
            error_rates[model_id] = rates
            
        return error_rates
