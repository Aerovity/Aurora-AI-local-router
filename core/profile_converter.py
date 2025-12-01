"""Convert adaptive_router profiles to mobile-optimized format.

This utility converts standard router profiles (from adaptive_router-main)
into lightweight mobile profiles suitable for on-device inference.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ProfileConverter:
    """Converter for router profiles to mobile format."""

    @staticmethod
    def convert_to_mobile(
        source_profile_path: Path,
        output_path: Path,
        compress: bool = True,
        use_float16: bool = True
    ) -> Dict[str, Any]:
        """Convert standard router profile to mobile format.

        Args:
            source_profile_path: Path to source profile JSON
            output_path: Path for output mobile profile
            compress: Whether to compress cluster centers
            use_float16: Whether to use float16 for cluster centers

        Returns:
            Mobile profile dictionary
        """
        logger.info(f"Converting profile: {source_profile_path} -> {output_path}")

        # Load source profile
        with open(source_profile_path, 'r') as f:
            source_profile = json.load(f)

        # Extract cluster centers
        cluster_centers = np.array(
            source_profile['cluster_centers']['cluster_centers'],
            dtype=np.float32
        )

        # Optionally convert to float16 for memory savings
        if use_float16:
            cluster_centers = cluster_centers.astype(np.float16)
            logger.info("Converted cluster centers to float16")

        # Create mobile profile structure
        mobile_profile = {
            'version': '1.0',
            'metadata': {
                'n_clusters': source_profile['cluster_centers']['n_clusters'],
                'feature_dim': source_profile['cluster_centers']['feature_dim'],
                'embedding_model': source_profile['metadata']['embedding_model'],
                'lambda_min': source_profile['metadata'].get('lambda_min', 0.0),
                'lambda_max': source_profile['metadata'].get('lambda_max', 2.0),
                'default_cost_preference': source_profile['metadata'].get(
                    'default_cost_preference', 0.5
                ),
                'silhouette_score': source_profile['metadata'].get('silhouette_score'),
            },
            'cluster_centers': {
                'n_clusters': source_profile['cluster_centers']['n_clusters'],
                'feature_dim': source_profile['cluster_centers']['feature_dim'],
                'cluster_centers': cluster_centers.tolist(),
                'dtype': 'float16' if use_float16 else 'float32'
            },
            'llm_profiles': source_profile['llm_profiles'],
            'models': source_profile.get('models', [])
        }

        # Calculate size reduction
        original_size = len(json.dumps(source_profile))
        mobile_size = len(json.dumps(mobile_profile))
        reduction_pct = (1 - mobile_size / original_size) * 100

        logger.info(
            f"Profile size: {original_size / 1024:.1f}KB -> {mobile_size / 1024:.1f}KB "
            f"({reduction_pct:.1f}% reduction)"
        )

        # Save mobile profile
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(mobile_profile, f, indent=2)

        logger.info(f"Mobile profile saved to {output_path}")

        return mobile_profile

    @staticmethod
    def create_cactus_profile(
        models_info: list,
        error_rates: Dict[str, list],
        cluster_centers: np.ndarray,
        embedding_model: str = "all-MiniLM-L6-v2",
        output_path: Optional[Path] = None,
        lambda_min: float = 0.0,
        lambda_max: float = 2.0,
        default_cost_preference: float = 0.5
    ) -> Dict[str, Any]:
        """Create a mobile router profile for Cactus models.

        Args:
            models_info: List of model info dicts with keys:
                - model_id: str
                - size_mb: float
                - avg_tokens_per_sec: float
                - context_size: int
                - capabilities: List[str]
            error_rates: Dict mapping model_id to list of per-cluster error rates
            cluster_centers: Numpy array of cluster centers (n_clusters × feature_dim)
            embedding_model: Name of embedding model used
            output_path: Optional path to save profile
            lambda_min: Minimum lambda value
            lambda_max: Maximum lambda value
            default_cost_preference: Default cost preference

        Returns:
            Profile dictionary
        """
        logger.info(f"Creating Cactus profile for {len(models_info)} models")

        n_clusters = cluster_centers.shape[0]
        feature_dim = cluster_centers.shape[1]

        profile = {
            'version': '1.0',
            'metadata': {
                'n_clusters': n_clusters,
                'feature_dim': feature_dim,
                'embedding_model': embedding_model,
                'lambda_min': lambda_min,
                'lambda_max': lambda_max,
                'default_cost_preference': default_cost_preference,
                'target': 'cactus_compute',
            },
            'cluster_centers': {
                'n_clusters': n_clusters,
                'feature_dim': feature_dim,
                'cluster_centers': cluster_centers.astype(np.float16).tolist(),
                'dtype': 'float16'
            },
            'llm_profiles': error_rates,
            'models': models_info
        }

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(profile, f, indent=2)

            profile_size = output_path.stat().st_size / 1024
            logger.info(f"Saved Cactus profile to {output_path} ({profile_size:.1f}KB)")

        return profile

    @staticmethod
    def validate_profile(profile_path: Path) -> bool:
        """Validate mobile profile format.

        Args:
            profile_path: Path to profile JSON

        Returns:
            True if valid, False otherwise
        """
        try:
            with open(profile_path, 'r') as f:
                profile = json.load(f)

            # Check required keys
            required_keys = ['version', 'metadata', 'cluster_centers', 'llm_profiles']
            for key in required_keys:
                if key not in profile:
                    logger.error(f"Missing required key: {key}")
                    return False

            # Check metadata
            metadata_keys = ['n_clusters', 'feature_dim', 'embedding_model']
            for key in metadata_keys:
                if key not in profile['metadata']:
                    logger.error(f"Missing metadata key: {key}")
                    return False

            # Check cluster centers
            centers = profile['cluster_centers']
            if centers['n_clusters'] != profile['metadata']['n_clusters']:
                logger.error("Cluster count mismatch")
                return False

            # Check error rates
            n_clusters = profile['metadata']['n_clusters']
            for model_id, rates in profile['llm_profiles'].items():
                if len(rates) != n_clusters:
                    logger.error(
                        f"Model {model_id} has {len(rates)} error rates, "
                        f"expected {n_clusters}"
                    )
                    return False

            logger.info(f"Profile {profile_path} is valid")
            return True

        except Exception as e:
            logger.error(f"Profile validation failed: {e}")
            return False

    @staticmethod
    def get_profile_stats(profile_path: Path) -> Dict[str, Any]:
        """Get statistics about a profile.

        Args:
            profile_path: Path to profile JSON

        Returns:
            Statistics dictionary
        """
        with open(profile_path, 'r') as f:
            profile = json.load(f)

        file_size_kb = profile_path.stat().st_size / 1024

        # Calculate cluster center memory
        centers = np.array(profile['cluster_centers']['cluster_centers'])
        center_memory_mb = centers.nbytes / (1024 * 1024)

        # Get model info
        n_models = len(profile['llm_profiles'])

        # Calculate average error rates
        all_error_rates = []
        for rates in profile['llm_profiles'].values():
            all_error_rates.extend(rates)
        avg_error_rate = np.mean(all_error_rates) if all_error_rates else 0.0

        stats = {
            'file_size_kb': file_size_kb,
            'n_clusters': profile['metadata']['n_clusters'],
            'feature_dim': profile['metadata']['feature_dim'],
            'n_models': n_models,
            'cluster_memory_mb': center_memory_mb,
            'avg_error_rate': avg_error_rate,
            'embedding_model': profile['metadata']['embedding_model'],
            'models': list(profile['llm_profiles'].keys())
        }

        return stats
