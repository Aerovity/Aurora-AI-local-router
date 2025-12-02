"""
Router Benchmark - Comprehensive evaluation of routing strategies.

Compares V1, V2, and baseline routers on various metrics.
"""

import json
import time
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict

import numpy as np

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

from ..config import CACTUS_MODELS, MODEL_BY_ID
from .baselines import BaselineRouters, BaselineResult


@dataclass
class BenchmarkResult:
    """Result metrics for a single routing strategy."""
    strategy_name: str
    avg_size_mb: float
    min_size_mb: float
    max_size_mb: float
    std_size_mb: float
    avg_tokens_per_sec: float
    savings_vs_largest: float  # % savings vs always largest
    latency_improvement: float  # Estimated % improvement
    battery_savings: float  # Estimated % battery savings
    routing_latency_ms: float  # Time for routing decision
    model_distribution: Dict[str, int] = field(default_factory=dict)
    cluster_distribution: Dict[int, int] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComparisonResult:
    """Comparison between two routing strategies."""
    strategy_a: str
    strategy_b: str
    size_difference_mb: float
    size_difference_pct: float
    agreement_pct: float  # % of samples with same model selection


class RouterBenchmark:
    """Comprehensive router benchmarking framework.
    
    Supports:
    - V1 Router (MiniLM)
    - V2 CACTUS Router (Nomic)
    - Baseline strategies
    - External LLM routers (Claude, Gemini)
    
    Example:
        >>> benchmark = RouterBenchmark()
        >>> benchmark.add_router("v1", router_v1.route)
        >>> benchmark.add_router("v2", router_v2.route)
        >>> results = benchmark.run(n_samples=100)
    """
    
    # Custom test prompts covering various complexity levels
    CUSTOM_PROMPTS = [
        # Simple
        ("Hi there!", "simple"),
        ("What is 2+2?", "simple"),
        ("Tell me a joke", "simple"),
        ("What color is the sky?", "simple"),
        
        # Medium
        ("Explain what photosynthesis is", "medium"),
        ("What caused World War I?", "medium"),
        ("How does a combustion engine work?", "medium"),
        ("What is the difference between DNA and RNA?", "medium"),
        
        # Complex
        ("Explain quantum entanglement and its computing implications", "complex"),
        ("Analyze Kant's categorical imperative", "complex"),
        ("Derive the Schwarzschild radius", "complex"),
        ("Implement a red-black tree in Python", "complex"),
        
        # Very Complex
        ("Design a distributed consensus algorithm for Byzantine faults", "very_complex"),
        ("Prove implications of Riemann hypothesis on prime distribution", "very_complex"),
        ("Analyze cryptocurrency impact on global monetary policy", "very_complex"),
        ("Write a compiler frontend for a simple language", "very_complex"),
    ]
    
    def __init__(self):
        """Initialize benchmark framework."""
        self.routers: Dict[str, Callable] = {}
        self.results: Dict[str, List[Any]] = {}
        
        # Add baseline routers
        self.add_router("always_largest", BaselineRouters.always_largest)
        self.add_router("always_smallest", BaselineRouters.always_smallest)
        self.add_router("random", BaselineRouters.random_selection)
        self.add_router("size_weighted", BaselineRouters.size_weighted_random)
        self.add_router("medium_model", BaselineRouters.medium_model)
        
    def add_router(self, name: str, route_fn: Callable):
        """Add a routing strategy to benchmark.
        
        Args:
            name: Strategy name
            route_fn: Function that takes (prompt, cost_preference) and returns result
        """
        self.routers[name] = route_fn
        self.results[name] = []
        
    def load_test_data(
        self,
        n_samples: int = 100,
        use_mmlu: bool = True,
        include_custom: bool = True
    ) -> List[tuple]:
        """Load test data for benchmarking.
        
        Args:
            n_samples: Number of MMLU samples
            use_mmlu: Whether to use MMLU data
            include_custom: Whether to include custom prompts
            
        Returns:
            List of (prompt, category) tuples
        """
        samples = []
        
        if include_custom:
            samples.extend(self.CUSTOM_PROMPTS)
            
        if use_mmlu and HAS_DATASETS:
            print(f"📚 Loading {n_samples} MMLU samples...")
            mmlu = load_dataset("cais/mmlu", "all", split="test")
            
            topics = [
                "abstract_algebra", "anatomy", "computer_security", "astronomy",
                "international_law", "marketing", "philosophy", "electrical_engineering"
            ]
            
            per_topic = max(1, n_samples // len(topics))
            
            for topic in topics:
                topic_samples = [s for s in mmlu if s["subject"] == topic]
                for s in topic_samples[:per_topic]:
                    samples.append((s['question'], topic))
                    
            samples = samples[:n_samples + len(self.CUSTOM_PROMPTS)]
            
        return samples
    
    def run(
        self,
        samples: Optional[List[tuple]] = None,
        n_samples: int = 100,
        cost_preference: float = 0.5,
        verbose: bool = True
    ) -> Dict[str, BenchmarkResult]:
        """Run benchmark on all registered routers.
        
        Args:
            samples: Optional pre-loaded samples
            n_samples: Number of MMLU samples if loading
            cost_preference: Cost preference to use
            verbose: Print progress
            
        Returns:
            Dict mapping strategy name to BenchmarkResult
        """
        if samples is None:
            samples = self.load_test_data(n_samples)
            
        if verbose:
            print(f"\n{'='*60}")
            print("🏃 RUNNING BENCHMARK")
            print(f"{'='*60}")
            print(f"  Samples: {len(samples)}")
            print(f"  Routers: {list(self.routers.keys())}")
            print(f"  Cost Preference: {cost_preference}")
            print()
            
        # Clear previous results
        for name in self.routers:
            self.results[name] = []
            
        # Run benchmark
        for i, (prompt, category) in enumerate(samples):
            if verbose and (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{len(samples)}")
                
            for name, route_fn in self.routers.items():
                start_time = time.time()
                
                try:
                    result = route_fn(prompt, cost_preference)
                    latency_ms = (time.time() - start_time) * 1000
                    
                    # Extract model info
                    if hasattr(result, 'model_id'):
                        model_id = result.model_id
                        size_mb = result.size_mb if hasattr(result, 'size_mb') else MODEL_BY_ID.get(model_id, CACTUS_MODELS[0]).size_mb
                        tokens_per_sec = result.tokens_per_sec if hasattr(result, 'tokens_per_sec') else 100
                        cluster_id = result.cluster_id if hasattr(result, 'cluster_id') else -1
                    elif isinstance(result, dict):
                        model_id = result.get('model_id', 'unknown')
                        size_mb = result.get('size_mb', 500)
                        tokens_per_sec = result.get('tokens_per_sec', 100)
                        cluster_id = result.get('cluster_id', -1)
                    else:
                        model_id = str(result)
                        size_mb = MODEL_BY_ID.get(model_id, CACTUS_MODELS[0]).size_mb
                        tokens_per_sec = 100
                        cluster_id = -1
                        
                    self.results[name].append({
                        'model_id': model_id,
                        'size_mb': size_mb,
                        'tokens_per_sec': tokens_per_sec,
                        'cluster_id': cluster_id,
                        'latency_ms': latency_ms,
                        'category': category
                    })
                    
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️ {name} error: {e}")
                    self.results[name].append({
                        'model_id': 'error',
                        'size_mb': 500,
                        'tokens_per_sec': 100,
                        'cluster_id': -1,
                        'latency_ms': 0,
                        'category': category
                    })
                    
        # Calculate metrics
        metrics = {}
        for name, results in self.results.items():
            metrics[name] = self._calculate_metrics(name, results)
            
        if verbose:
            self._print_results(metrics)
            
        return metrics
    
    def _calculate_metrics(self, name: str, results: List[dict]) -> BenchmarkResult:
        """Calculate metrics for a routing strategy."""
        sizes = [r['size_mb'] for r in results]
        speeds = [r['tokens_per_sec'] for r in results]
        latencies = [r['latency_ms'] for r in results]
        
        # Size statistics
        avg_size = statistics.mean(sizes) if sizes else 0
        min_size = min(sizes) if sizes else 0
        max_size = max(sizes) if sizes else 0
        std_size = statistics.stdev(sizes) if len(sizes) > 1 else 0
        
        # Savings vs largest
        largest_size = 1440  # lfm2-vl-1.6b
        savings = ((largest_size - avg_size) / largest_size) * 100
        
        # Estimated improvements
        latency_improvement = savings * 0.8
        battery_savings = savings * 0.7
        
        # Distributions
        model_dist = defaultdict(int)
        cluster_dist = defaultdict(int)
        
        for r in results:
            model_dist[r['model_id']] += 1
            if r['cluster_id'] >= 0:
                cluster_dist[r['cluster_id']] += 1
                
        return BenchmarkResult(
            strategy_name=name,
            avg_size_mb=avg_size,
            min_size_mb=min_size,
            max_size_mb=max_size,
            std_size_mb=std_size,
            avg_tokens_per_sec=statistics.mean(speeds) if speeds else 0,
            savings_vs_largest=savings,
            latency_improvement=latency_improvement,
            battery_savings=battery_savings,
            routing_latency_ms=statistics.mean(latencies) if latencies else 0,
            model_distribution=dict(model_dist),
            cluster_distribution=dict(cluster_dist)
        )
    
    def _print_results(self, metrics: Dict[str, BenchmarkResult]):
        """Print benchmark results."""
        print(f"\n{'='*80}")
        print("📊 BENCHMARK RESULTS")
        print(f"{'='*80}\n")
        
        # Summary table
        print(f"{'Strategy':<18} {'Avg Size':<10} {'Savings':<10} {'Latency↓':<10} {'Battery↓':<10} {'Route ms'}")
        print("-" * 80)
        
        for name, m in sorted(metrics.items(), key=lambda x: -x[1].savings_vs_largest):
            print(f"{name:<18} {m.avg_size_mb:>7.0f}MB  {m.savings_vs_largest:>7.1f}%  "
                  f"{m.latency_improvement:>7.1f}%  {m.battery_savings:>7.1f}%  "
                  f"{m.routing_latency_ms:>7.2f}")
                  
        print()
        
    def compare(
        self,
        strategy_a: str,
        strategy_b: str
    ) -> ComparisonResult:
        """Compare two routing strategies.
        
        Args:
            strategy_a: First strategy name
            strategy_b: Second strategy name
            
        Returns:
            ComparisonResult with comparison metrics
        """
        results_a = self.results.get(strategy_a, [])
        results_b = self.results.get(strategy_b, [])
        
        if not results_a or not results_b:
            raise ValueError(f"Missing results for {strategy_a} or {strategy_b}")
            
        # Calculate agreement
        agreements = sum(
            1 for a, b in zip(results_a, results_b) 
            if a['model_id'] == b['model_id']
        )
        agreement_pct = (agreements / len(results_a)) * 100
        
        # Size difference
        avg_a = statistics.mean([r['size_mb'] for r in results_a])
        avg_b = statistics.mean([r['size_mb'] for r in results_b])
        
        return ComparisonResult(
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            size_difference_mb=avg_a - avg_b,
            size_difference_pct=((avg_a - avg_b) / avg_b) * 100 if avg_b > 0 else 0,
            agreement_pct=agreement_pct
        )
    
    def save_results(self, output_path: Path):
        """Save benchmark results to JSON.
        
        Args:
            output_path: Path to save results
        """
        output_path = Path(output_path)
        
        # Prepare results for serialization
        output = {
            'results': {},
            'raw_data': {}
        }
        
        for name, results in self.results.items():
            metrics = self._calculate_metrics(name, results)
            output['results'][name] = metrics.to_dict()
            output['raw_data'][name] = results
            
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
            
        print(f"✅ Results saved to {output_path}")
