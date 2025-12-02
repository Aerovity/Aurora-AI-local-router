#!/usr/bin/env python3
"""
AuroraAI Router - Main CLI Entry Point

Unified command-line interface for training, benchmarking, and routing.

Usage:
    python -m auroraai_router train --version v2 --output profiles/my_profile.json
    python -m auroraai_router benchmark --profile profiles/cactus_profile.json
    python -m auroraai_router route --profile profiles/cactus_profile.json "What is quantum entanglement?"
    python -m auroraai_router compare --v1 profiles/v1.json --v2 profiles/cactus.json
"""

import argparse
import sys
from pathlib import Path


def cmd_train(args):
    """Train a new router profile."""
    if args.version == 'v1':
        from .training import TrainerV1
        trainer = TrainerV1()
    else:
        from .training import CactusTrainer
        trainer = CactusTrainer()
    
    print(f"Training {args.version.upper()} router...")
    
    # Run training
    profile_path = trainer.train(
        output_path=args.output,
        n_samples=args.samples,
        use_gpu=not args.cpu
    )
    
    print(f"Profile saved to: {profile_path}")


def cmd_benchmark(args):
    """Benchmark a router profile."""
    from .benchmarks import RouterBenchmark
    from .router_factory import create_router
    
    # Load router
    router = create_router(profile_path=args.profile)
    
    # Run benchmark
    benchmark = RouterBenchmark()
    results = benchmark.run_benchmark(
        router=router,
        test_data=None,  # Will load MMLU
        n_samples=args.samples
    )
    
    print(results.summary())
    
    if args.output:
        benchmark.save_results(results, args.output)


def cmd_compare(args):
    """Compare V1 and V2 routers."""
    from .benchmarks import compare_routers
    
    result = compare_routers(
        v1_profile=args.v1,
        v2_profile=args.v2,
        output_dir=args.output,
        visualize=not args.no_viz
    )
    
    print(result.summary())


def cmd_route(args):
    """Route a single query."""
    from .router_factory import create_router
    
    router = create_router(profile_path=args.profile)
    
    query = " ".join(args.query)
    result = router.route(query)
    
    print(f"\nQuery: {query}")
    print(f"Selected Model: {result['model']}")
    print(f"Confidence: {result.get('confidence', 'N/A')}")
    print(f"Cluster: {result.get('cluster', 'N/A')}")
    
    if args.verbose:
        print(f"\nAll Scores:")
        for model, score in sorted(result.get('all_scores', {}).items(), 
                                   key=lambda x: x[1], reverse=True):
            print(f"  {model}: {score:.4f}")


def cmd_info(args):
    """Show information about a profile."""
    import json
    
    with open(args.profile, 'r') as f:
        profile = json.load(f)
    
    print(f"\nProfile: {args.profile}")
    print("=" * 50)
    
    # Detect version
    if 'cluster_assignments' in profile:
        version = 'V2 (CACTUS/Nomic)'
        n_clusters = len(set(profile['cluster_assignments']))
        embedding_dim = len(profile['cluster_centers'][0]) if profile.get('cluster_centers') else 'Unknown'
    else:
        version = 'V1 (MiniLM/KMeans)'
        n_clusters = profile.get('n_clusters', 'Unknown')
        embedding_dim = 384
    
    print(f"Version: {version}")
    print(f"Embedding Dimension: {embedding_dim}")
    print(f"Number of Clusters: {n_clusters}")
    
    if 'models' in profile:
        print(f"Models: {len(profile['models'])}")
        for model in profile['models'][:5]:
            name = model.get('name', model.get('id', 'unknown'))
            size = model.get('size_mb', 'N/A')
            print(f"  - {name} ({size}MB)")
        if len(profile['models']) > 5:
            print(f"  ... and {len(profile['models']) - 5} more")
    
    if 'metadata' in profile:
        meta = profile['metadata']
        print(f"\nMetadata:")
        print(f"  Created: {meta.get('created_at', 'Unknown')}")
        print(f"  Training Samples: {meta.get('n_samples', 'Unknown')}")
        print(f"  Silhouette Score: {meta.get('silhouette_score', 'Unknown')}")


def main():
    parser = argparse.ArgumentParser(
        description="AuroraAI Router - Intelligent LLM Routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new V2 profile
  python -m auroraai_router train --version v2 --output my_profile.json

  # Benchmark a profile
  python -m auroraai_router benchmark --profile profiles/cactus_profile.json

  # Compare V1 and V2
  python -m auroraai_router compare --v2 profiles/cactus_profile.json

  # Route a query
  python -m auroraai_router route --profile profiles/cactus_profile.json "What is gravity?"

  # Show profile info
  python -m auroraai_router info --profile profiles/cactus_profile.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new router profile')
    train_parser.add_argument('--version', choices=['v1', 'v2'], default='v2',
                             help='Router version to train (default: v2)')
    train_parser.add_argument('--output', '-o', type=str, default='profiles/profile.json',
                             help='Output path for the profile')
    train_parser.add_argument('--samples', '-n', type=int, default=2000,
                             help='Number of training samples')
    train_parser.add_argument('--cpu', action='store_true',
                             help='Force CPU-only training')
    train_parser.set_defaults(func=cmd_train)
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark a router profile')
    bench_parser.add_argument('--profile', '-p', type=str, required=True,
                             help='Path to router profile')
    bench_parser.add_argument('--samples', '-n', type=int, default=500,
                             help='Number of test samples')
    bench_parser.add_argument('--output', '-o', type=str,
                             help='Output path for results JSON')
    bench_parser.set_defaults(func=cmd_benchmark)
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare V1 and V2 routers')
    compare_parser.add_argument('--v1', type=str,
                               help='Path to V1 profile (optional)')
    compare_parser.add_argument('--v2', type=str, required=True,
                               help='Path to V2 profile')
    compare_parser.add_argument('--output', '-o', type=str, default='comparison_results',
                               help='Output directory')
    compare_parser.add_argument('--no-viz', action='store_true',
                               help='Skip visualization')
    compare_parser.set_defaults(func=cmd_compare)
    
    # Route command
    route_parser = subparsers.add_parser('route', help='Route a single query')
    route_parser.add_argument('--profile', '-p', type=str, required=True,
                             help='Path to router profile')
    route_parser.add_argument('--verbose', '-v', action='store_true',
                             help='Show detailed scores')
    route_parser.add_argument('query', nargs='+', help='Query to route')
    route_parser.set_defaults(func=cmd_route)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show profile information')
    info_parser.add_argument('--profile', '-p', type=str, required=True,
                            help='Path to router profile')
    info_parser.set_defaults(func=cmd_info)
    
    # Parse and execute
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()
