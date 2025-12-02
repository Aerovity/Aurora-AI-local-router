#!/usr/bin/env python3
"""
Example: Benchmark Comparison

Demonstrates how to compare V1 and V2 routers using the benchmarking module.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks import compare_routers, RouterBenchmark
from src import create_router
from src.config import PathConfig


def run_comparison():
    """Run a full V1 vs V2 comparison."""
    
    print("=" * 60)
    print("AuroraAI Router - Benchmark Comparison Example")
    print("=" * 60)
    
    v2_profile = PathConfig.PROFILE_PATH
    
    if not v2_profile.exists():
        print(f"ERROR: V2 profile not found at {v2_profile}")
        return
    
    print(f"\nV2 Profile: {v2_profile}")
    print("V1 Profile: Not available (will use baselines for comparison)")
    
    # Run comparison
    result = compare_routers(
        v2_profile=str(v2_profile),
        output_dir="benchmark_results",
        visualize=True  # Set to False if matplotlib not available
    )
    
    print("\nComparison complete!")
    print(f"Results saved to: benchmark_results/")


def run_single_benchmark():
    """Run benchmark on single router."""
    
    print("=" * 60)
    print("AuroraAI Router - Single Router Benchmark")
    print("=" * 60)
    
    profile_path = PathConfig.PROFILE_PATH
    
    if not profile_path.exists():
        print(f"ERROR: Profile not found at {profile_path}")
        return
    
    # Load router
    router = create_router(profile_path)
    
    # Create benchmark
    benchmark = RouterBenchmark()
    
    # Generate synthetic test data (replace with real data)
    test_data = [
        {"question": "What is 2+2?", "topic": "math"},
        {"question": "Define photosynthesis", "topic": "biology"},
        {"question": "Who wrote Hamlet?", "topic": "literature"},
        {"question": "Explain quantum mechanics", "topic": "physics"},
        {"question": "What is machine learning?", "topic": "computer_science"},
    ] * 20  # Repeat for more samples
    
    print(f"\nRunning benchmark with {len(test_data)} samples...")
    
    result = benchmark.run_benchmark(router, test_data)
    
    print("\nResults:")
    print(result.summary())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Run comparison")
    parser.add_argument("--single", action="store_true", help="Run single benchmark")
    
    args = parser.parse_args()
    
    if args.compare:
        run_comparison()
    elif args.single:
        run_single_benchmark()
    else:
        # Default: run single benchmark
        run_single_benchmark()
