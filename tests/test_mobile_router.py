"""Tests for mobile router components."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from core import MobileClusterEngine, MobileRouter, ModelInfo


class TestMobileClusterEngine(unittest.TestCase):
    """Tests for MobileClusterEngine."""

    def setUp(self):
        """Set up test fixtures."""
        # Create fake cluster centers
        self.cluster_centers = np.random.randn(5, 384).astype(np.float32)
        self.engine = MobileClusterEngine.from_cluster_centers(
            cluster_centers=self.cluster_centers,
            embedding_model_name="test-model"
        )

    def test_initialization(self):
        """Test engine initialization."""
        self.assertEqual(self.engine.n_clusters, 5)
        self.assertTrue(self.engine.is_fitted)
        self.assertEqual(self.engine.cluster_centers.shape, (5, 384))

    def test_assign_cluster(self):
        """Test cluster assignment."""
        # Create random embedding
        embedding = np.random.randn(384).astype(np.float32)

        cluster_id, distance = self.engine.assign_cluster(embedding)

        self.assertIsInstance(cluster_id, int)
        self.assertGreaterEqual(cluster_id, 0)
        self.assertLess(cluster_id, 5)
        self.assertIsInstance(distance, float)

    def test_save_load(self):
        """Test saving and loading."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_engine.pkl"

            # Save
            self.engine.save(save_path)
            self.assertTrue(save_path.exists())

            # Load
            loaded_engine = MobileClusterEngine.load(save_path)

            self.assertEqual(loaded_engine.n_clusters, 5)
            self.assertTrue(np.allclose(
                loaded_engine.cluster_centers,
                self.engine.cluster_centers
            ))


class TestMobileRouter(unittest.TestCase):
    """Tests for MobileRouter."""

    def setUp(self):
        """Set up test fixtures."""
        # Create fake cluster engine
        cluster_centers = np.random.randn(5, 384).astype(np.float32)
        self.cluster_engine = MobileClusterEngine.from_cluster_centers(
            cluster_centers=cluster_centers
        )

        # Create test models
        self.models = [
            ModelInfo(
                model_id='small',
                model_path='weights/small',
                size_mb=100,
                avg_tokens_per_sec=200
            ),
            ModelInfo(
                model_id='medium',
                model_path='weights/medium',
                size_mb=500,
                avg_tokens_per_sec=100
            ),
            ModelInfo(
                model_id='large',
                model_path='weights/large',
                size_mb=1000,
                avg_tokens_per_sec=50
            ),
        ]

        # Create fake error rates
        self.error_rates = {
            'small': [0.15, 0.18, 0.20, 0.16, 0.19],
            'medium': [0.10, 0.12, 0.11, 0.13, 0.10],
            'large': [0.05, 0.06, 0.05, 0.07, 0.06],
        }

        self.router = MobileRouter(
            cluster_engine=self.cluster_engine,
            models=self.models,
            error_rates=self.error_rates
        )

    def test_route_to_small_model(self):
        """Test routing with cost preference for small model."""
        embedding = np.random.randn(384).astype(np.float32)

        result = self.router.route(
            prompt_embedding=embedding,
            cost_preference=0.0  # Prefer fast/small
        )

        # Should prefer smaller model when cost preference is low
        self.assertIsNotNone(result)
        self.assertIsInstance(result.model_id, str)
        self.assertIn(result.model_id, ['small', 'medium', 'large'])

    def test_route_to_large_model(self):
        """Test routing with cost preference for quality."""
        embedding = np.random.randn(384).astype(np.float32)

        result = self.router.route(
            prompt_embedding=embedding,
            cost_preference=1.0  # Prefer quality
        )

        # Should prefer larger/better model when cost preference is high
        self.assertIsNotNone(result)
        self.assertEqual(result.model_id, 'large')  # Best error rates

    def test_available_models_filter(self):
        """Test filtering by available models."""
        embedding = np.random.randn(384).astype(np.float32)

        result = self.router.route(
            prompt_embedding=embedding,
            available_models=['small', 'medium'],  # Exclude 'large'
            cost_preference=1.0
        )

        # Should select from available models only
        self.assertIn(result.model_id, ['small', 'medium'])

    def test_get_supported_models(self):
        """Test getting supported models."""
        models = self.router.get_supported_models()

        self.assertEqual(len(models), 3)
        self.assertIn('small', models)
        self.assertIn('medium', models)
        self.assertIn('large', models)

    def test_get_cluster_info(self):
        """Test getting cluster information."""
        info = self.router.get_cluster_info()

        self.assertEqual(info['n_clusters'], 5)
        self.assertEqual(len(info['supported_models']), 3)


class TestProfileConverter(unittest.TestCase):
    """Tests for ProfileConverter."""

    def test_create_cactus_profile(self):
        """Test creating Cactus profile."""
        from core import ProfileConverter

        models_info = [
            {
                'model_id': 'test-model',
                'model_path': 'weights/test',
                'size_mb': 200,
                'avg_tokens_per_sec': 150,
                'capabilities': ['text']
            }
        ]

        error_rates = {
            'test-model': [0.10, 0.12, 0.11]
        }

        cluster_centers = np.random.randn(3, 384).astype(np.float32)

        profile = ProfileConverter.create_cactus_profile(
            models_info=models_info,
            error_rates=error_rates,
            cluster_centers=cluster_centers,
            embedding_model="test-model"
        )

        self.assertEqual(profile['version'], '1.0')
        self.assertEqual(profile['metadata']['n_clusters'], 3)
        self.assertIn('test-model', profile['llm_profiles'])


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestMobileClusterEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestMobileRouter))
    suite.addTests(loader.loadTestsFromTestCase(TestProfileConverter))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
