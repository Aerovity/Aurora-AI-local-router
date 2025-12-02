"""
Router Comparison Module

Compares V1 (MiniLM/KMeans) vs V2 (Nomic/HDBSCAN) router performance
with comprehensive visualizations and statistical analysis.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..config import CACTUS_MODELS, PathConfig
from ..router_factory import create_router, RouterType
from .benchmark import RouterBenchmark, BenchmarkResult
from .baselines import create_all_baselines


@dataclass
class ComparisonResult:
    """Results from comparing two routers."""
    v1_results: BenchmarkResult
    v2_results: BenchmarkResult
    baseline_results: Dict[str, BenchmarkResult]
    
    # Comparison metrics
    accuracy_delta: float = 0.0
    latency_delta: float = 0.0
    efficiency_delta: float = 0.0
    
    # Statistical significance
    accuracy_pvalue: Optional[float] = None
    is_significant: bool = False
    
    # Per-topic comparison
    topic_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate comparison metrics."""
        self.accuracy_delta = self.v2_results.accuracy - self.v1_results.accuracy
        self.latency_delta = self.v2_results.avg_latency_ms - self.v1_results.avg_latency_ms
        self.efficiency_delta = self.v2_results.efficiency_score - self.v1_results.efficiency_score
    
    def summary(self) -> str:
        """Generate comparison summary."""
        lines = [
            "=" * 60,
            "ROUTER COMPARISON: V1 (MiniLM) vs V2 (CACTUS/Nomic)",
            "=" * 60,
            "",
            "ACCURACY COMPARISON:",
            f"  V1 Router: {self.v1_results.accuracy:.2%}",
            f"  V2 Router: {self.v2_results.accuracy:.2%}",
            f"  Delta:     {self.accuracy_delta:+.2%} {'✓ V2 better' if self.accuracy_delta > 0 else '✗ V1 better'}",
            "",
            "LATENCY COMPARISON:",
            f"  V1 Router: {self.v1_results.avg_latency_ms:.2f}ms",
            f"  V2 Router: {self.v2_results.avg_latency_ms:.2f}ms",
            f"  Delta:     {self.latency_delta:+.2f}ms {'✓ V1 faster' if self.latency_delta > 0 else '✓ V2 faster'}",
            "",
            "EFFICIENCY COMPARISON:",
            f"  V1 Router: {self.v1_results.efficiency_score:.4f}",
            f"  V2 Router: {self.v2_results.efficiency_score:.4f}",
            f"  Delta:     {self.efficiency_delta:+.4f}",
            "",
        ]
        
        if self.accuracy_pvalue is not None:
            lines.extend([
                "STATISTICAL SIGNIFICANCE:",
                f"  p-value: {self.accuracy_pvalue:.4f}",
                f"  Significant (p<0.05): {'Yes' if self.is_significant else 'No'}",
                "",
            ])
        
        # Baseline comparison
        lines.extend([
            "BASELINE COMPARISON:",
        ])
        for name, result in self.baseline_results.items():
            v2_vs_baseline = self.v2_results.accuracy - result.accuracy
            lines.append(f"  vs {name}: {v2_vs_baseline:+.2%}")
        
        lines.extend([
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)


class RouterComparison:
    """
    Comprehensive comparison between V1 and V2 routers.
    
    Example:
        comparison = RouterComparison(
            v1_profile_path="profiles/v1_profile.json",
            v2_profile_path="profiles/cactus_profile.json"
        )
        results = comparison.run_comparison(test_data)
        comparison.visualize(results)
    """
    
    def __init__(
        self,
        v1_profile_path: Optional[str] = None,
        v2_profile_path: Optional[str] = None,
        models: List[Dict] = None
    ):
        """
        Initialize comparison.
        
        Args:
            v1_profile_path: Path to V1 profile (optional)
            v2_profile_path: Path to V2 profile
            models: List of model configs
        """
        self.models = models or CACTUS_MODELS
        
        # Create routers
        self.v1_router = None
        self.v2_router = None
        
        if v1_profile_path and Path(v1_profile_path).exists():
            self.v1_router = create_router(
                profile_path=v1_profile_path,
                router_type=RouterType.V1
            )
        
        if v2_profile_path and Path(v2_profile_path).exists():
            self.v2_router = create_router(
                profile_path=v2_profile_path,
                router_type=RouterType.V2
            )
        
        # Create baselines
        self.baselines = create_all_baselines(self.models)
        
        # Benchmark instance
        self.benchmark = RouterBenchmark(models=self.models)
    
    def run_comparison(
        self,
        test_data: List[Dict],
        n_runs: int = 3
    ) -> ComparisonResult:
        """
        Run comprehensive comparison.
        
        Args:
            test_data: List of test examples with 'question' and 'topic'
            n_runs: Number of runs for statistical significance
            
        Returns:
            ComparisonResult with all comparison data
        """
        print("Running Router Comparison...")
        print("=" * 50)
        
        # Run V1 benchmark
        v1_results = None
        if self.v1_router:
            print("\n[1/4] Benchmarking V1 Router (MiniLM/KMeans)...")
            v1_results = self.benchmark.run_benchmark(self.v1_router, test_data)
        else:
            print("\n[1/4] V1 Router not available, creating placeholder...")
            v1_results = BenchmarkResult(
                router_name="V1 Router (Not Available)",
                total_samples=len(test_data),
                correct_predictions=0,
                accuracy=0.0,
                avg_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                model_distribution={},
                topic_accuracy={},
                efficiency_score=0.0
            )
        
        # Run V2 benchmark
        v2_results = None
        if self.v2_router:
            print("\n[2/4] Benchmarking V2 Router (Nomic/HDBSCAN)...")
            v2_results = self.benchmark.run_benchmark(self.v2_router, test_data)
        else:
            print("\n[2/4] V2 Router not available, creating placeholder...")
            v2_results = BenchmarkResult(
                router_name="V2 Router (Not Available)",
                total_samples=len(test_data),
                correct_predictions=0,
                accuracy=0.0,
                avg_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                model_distribution={},
                topic_accuracy={},
                efficiency_score=0.0
            )
        
        # Run baseline benchmarks
        print("\n[3/4] Benchmarking Baselines...")
        baseline_results = {}
        for name, baseline in self.baselines.items():
            print(f"  - {name}...")
            baseline_results[name] = self.benchmark.run_benchmark(baseline, test_data)
        
        # Statistical significance testing
        print("\n[4/4] Computing Statistical Significance...")
        pvalue = None
        is_significant = False
        
        if HAS_SCIPY and self.v1_router and self.v2_router and n_runs > 1:
            v1_accuracies = []
            v2_accuracies = []
            
            for _ in range(n_runs):
                v1_acc = self.benchmark.run_benchmark(self.v1_router, test_data).accuracy
                v2_acc = self.benchmark.run_benchmark(self.v2_router, test_data).accuracy
                v1_accuracies.append(v1_acc)
                v2_accuracies.append(v2_acc)
            
            # Paired t-test
            _, pvalue = stats.ttest_rel(v1_accuracies, v2_accuracies)
            is_significant = pvalue < 0.05
        
        # Create comparison result
        result = ComparisonResult(
            v1_results=v1_results,
            v2_results=v2_results,
            baseline_results=baseline_results,
            accuracy_pvalue=pvalue,
            is_significant=is_significant
        )
        
        # Topic comparison
        if self.v1_router and self.v2_router:
            all_topics = set(v1_results.topic_accuracy.keys()) | set(v2_results.topic_accuracy.keys())
            for topic in all_topics:
                v1_acc = v1_results.topic_accuracy.get(topic, 0.0)
                v2_acc = v2_results.topic_accuracy.get(topic, 0.0)
                result.topic_comparison[topic] = {
                    'v1': v1_acc,
                    'v2': v2_acc,
                    'delta': v2_acc - v1_acc
                }
        
        print("\n" + result.summary())
        return result
    
    def visualize(
        self,
        result: ComparisonResult,
        output_path: Optional[str] = None
    ):
        """
        Create visualization of comparison results.
        
        Args:
            result: ComparisonResult from run_comparison
            output_path: Path to save figure (optional)
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Accuracy Comparison Bar Chart
        ax1 = axes[0, 0]
        routers = ['V1\n(MiniLM)', 'V2\n(Nomic)']
        accuracies = [result.v1_results.accuracy, result.v2_results.accuracy]
        
        # Add baselines
        for name, baseline_result in result.baseline_results.items():
            routers.append(name.replace('Router', ''))
            accuracies.append(baseline_result.accuracy)
        
        colors = ['#3498db', '#e74c3c'] + ['#95a5a6'] * len(result.baseline_results)
        bars = ax1.bar(routers, accuracies, color=colors, edgecolor='black')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Router Accuracy Comparison')
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.1%}', ha='center', va='bottom', fontsize=9)
        
        # 2. Latency Comparison
        ax2 = axes[0, 1]
        latency_data = {
            'V1': [result.v1_results.p50_latency_ms, result.v1_results.p95_latency_ms, result.v1_results.p99_latency_ms],
            'V2': [result.v2_results.p50_latency_ms, result.v2_results.p95_latency_ms, result.v2_results.p99_latency_ms]
        }
        
        x = np.arange(3)
        width = 0.35
        
        ax2.bar(x - width/2, latency_data['V1'], width, label='V1 (MiniLM)', color='#3498db')
        ax2.bar(x + width/2, latency_data['V2'], width, label='V2 (Nomic)', color='#e74c3c')
        
        ax2.set_ylabel('Latency (ms)')
        ax2.set_title('Latency Percentiles')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['P50', 'P95', 'P99'])
        ax2.legend()
        
        # 3. Topic-wise Accuracy Comparison
        ax3 = axes[1, 0]
        if result.topic_comparison:
            topics = list(result.topic_comparison.keys())[:10]  # Top 10 topics
            v1_accs = [result.topic_comparison[t]['v1'] for t in topics]
            v2_accs = [result.topic_comparison[t]['v2'] for t in topics]
            
            x = np.arange(len(topics))
            width = 0.35
            
            ax3.barh(x - width/2, v1_accs, width, label='V1', color='#3498db')
            ax3.barh(x + width/2, v2_accs, width, label='V2', color='#e74c3c')
            
            ax3.set_xlabel('Accuracy')
            ax3.set_title('Per-Topic Accuracy')
            ax3.set_yticks(x)
            ax3.set_yticklabels([t[:15] + '...' if len(t) > 15 else t for t in topics], fontsize=8)
            ax3.legend()
            ax3.set_xlim(0, 1)
        
        # 4. Model Distribution Comparison
        ax4 = axes[1, 1]
        
        v1_dist = result.v1_results.model_distribution
        v2_dist = result.v2_results.model_distribution
        
        all_models = sorted(set(v1_dist.keys()) | set(v2_dist.keys()))
        
        if all_models:
            v1_counts = [v1_dist.get(m, 0) for m in all_models]
            v2_counts = [v2_dist.get(m, 0) for m in all_models]
            
            x = np.arange(len(all_models))
            width = 0.35
            
            ax4.bar(x - width/2, v1_counts, width, label='V1', color='#3498db')
            ax4.bar(x + width/2, v2_counts, width, label='V2', color='#e74c3c')
            
            ax4.set_ylabel('Selection Count')
            ax4.set_title('Model Selection Distribution')
            ax4.set_xticks(x)
            ax4.set_xticklabels([m[:10] for m in all_models], rotation=45, ha='right', fontsize=8)
            ax4.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison visualization to {output_path}")
        
        plt.show()
    
    def generate_report(
        self,
        result: ComparisonResult,
        output_path: str
    ):
        """
        Generate detailed comparison report as JSON.
        
        Args:
            result: ComparisonResult
            output_path: Path to save JSON report
        """
        report = {
            'summary': {
                'v1_accuracy': result.v1_results.accuracy,
                'v2_accuracy': result.v2_results.accuracy,
                'accuracy_improvement': result.accuracy_delta,
                'latency_change_ms': result.latency_delta,
                'statistically_significant': result.is_significant,
                'p_value': result.accuracy_pvalue
            },
            'v1_details': {
                'router_name': result.v1_results.router_name,
                'total_samples': result.v1_results.total_samples,
                'correct': result.v1_results.correct_predictions,
                'accuracy': result.v1_results.accuracy,
                'avg_latency_ms': result.v1_results.avg_latency_ms,
                'p50_latency_ms': result.v1_results.p50_latency_ms,
                'p95_latency_ms': result.v1_results.p95_latency_ms,
                'efficiency_score': result.v1_results.efficiency_score,
                'model_distribution': result.v1_results.model_distribution,
                'topic_accuracy': result.v1_results.topic_accuracy
            },
            'v2_details': {
                'router_name': result.v2_results.router_name,
                'total_samples': result.v2_results.total_samples,
                'correct': result.v2_results.correct_predictions,
                'accuracy': result.v2_results.accuracy,
                'avg_latency_ms': result.v2_results.avg_latency_ms,
                'p50_latency_ms': result.v2_results.p50_latency_ms,
                'p95_latency_ms': result.v2_results.p95_latency_ms,
                'efficiency_score': result.v2_results.efficiency_score,
                'model_distribution': result.v2_results.model_distribution,
                'topic_accuracy': result.v2_results.topic_accuracy
            },
            'baselines': {
                name: {
                    'accuracy': r.accuracy,
                    'avg_latency_ms': r.avg_latency_ms,
                    'model_distribution': r.model_distribution
                }
                for name, r in result.baseline_results.items()
            },
            'topic_comparison': result.topic_comparison
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Saved detailed report to {output_path}")


def compare_routers(
    v1_profile: Optional[str] = None,
    v2_profile: Optional[str] = None,
    test_data: Optional[List[Dict]] = None,
    output_dir: str = "comparison_results",
    visualize: bool = True
) -> ComparisonResult:
    """
    Convenience function to run full router comparison.
    
    Args:
        v1_profile: Path to V1 profile
        v2_profile: Path to V2 profile
        test_data: Test data (loads MMLU if not provided)
        output_dir: Directory for outputs
        visualize: Whether to create visualizations
        
    Returns:
        ComparisonResult
    """
    # Default paths
    if v2_profile is None:
        v2_profile = PathConfig.PROFILE_PATH
    
    # Load test data if not provided
    if test_data is None:
        try:
            from datasets import load_dataset
            print("Loading MMLU test data...")
            dataset = load_dataset("cais/mmlu", "all", split="test")
            test_data = [
                {'question': row['question'], 'topic': row['subject']}
                for row in list(dataset)[:500]  # Sample for comparison
            ]
        except Exception as e:
            print(f"Could not load MMLU: {e}")
            # Create synthetic test data
            test_data = [
                {'question': f"Test question {i}", 'topic': 'test'}
                for i in range(100)
            ]
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Run comparison
    comparison = RouterComparison(
        v1_profile_path=v1_profile,
        v2_profile_path=v2_profile
    )
    
    result = comparison.run_comparison(test_data)
    
    # Generate outputs
    comparison.generate_report(result, str(output_dir / "comparison_report.json"))
    
    if visualize and HAS_MATPLOTLIB:
        comparison.visualize(result, str(output_dir / "comparison_charts.png"))
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare V1 and V2 routers")
    parser.add_argument("--v1-profile", type=str, help="Path to V1 profile")
    parser.add_argument("--v2-profile", type=str, default=str(PathConfig.PROFILE_PATH),
                       help="Path to V2 profile")
    parser.add_argument("--output-dir", type=str, default="comparison_results",
                       help="Output directory")
    parser.add_argument("--no-visualize", action="store_true",
                       help="Skip visualization")
    
    args = parser.parse_args()
    
    result = compare_routers(
        v1_profile=args.v1_profile,
        v2_profile=args.v2_profile,
        output_dir=args.output_dir,
        visualize=not args.no_visualize
    )
