"""
Aurora AI Router - Benchmarking Module

Provides comprehensive benchmarking tools for evaluating router performance.
"""

from .baselines import (
    BaselineRouter,
    AlwaysLargestRouter,
    AlwaysSmallestRouter,
    RandomRouter,
    RoundRobinRouter,
    create_all_baselines
)
from .benchmark import (
    BenchmarkResult,
    RouterBenchmark
)
from .compare import (
    ComparisonResult,
    RouterComparison,
    compare_routers
)

__all__ = [
    # Baselines
    'BaselineRouter',
    'AlwaysLargestRouter',
    'AlwaysSmallestRouter',
    'RandomRouter',
    'RoundRobinRouter',
    'create_all_baselines',
    
    # Benchmark
    'BenchmarkResult',
    'RouterBenchmark',
    
    # Comparison
    'ComparisonResult',
    'RouterComparison',
    'compare_routers',
]
