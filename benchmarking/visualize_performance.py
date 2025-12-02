#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize how performance metrics are calculated.
Shows step-by-step how we get:
- Average model size
- Savings vs largest
- Latency improvement
- Battery savings
"""

import sys
import io
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Model data
CACTUS_MODELS = [
    {'model_id': 'gemma-270m', 'size_mb': 172, 'avg_tokens_per_sec': 173},
    {'model_id': 'smollm-360m', 'size_mb': 227, 'avg_tokens_per_sec': 150},
    {'model_id': 'lfm2-350m', 'size_mb': 233, 'avg_tokens_per_sec': 145},
    {'model_id': 'qwen-600m', 'size_mb': 394, 'avg_tokens_per_sec': 129},
    {'model_id': 'lfm2-vl-450m', 'size_mb': 420, 'avg_tokens_per_sec': 113},
    {'model_id': 'lfm2-700m', 'size_mb': 467, 'avg_tokens_per_sec': 115},
    {'model_id': 'gemma-1b', 'size_mb': 642, 'avg_tokens_per_sec': 100},
    {'model_id': 'lfm2-1.2b', 'size_mb': 722, 'avg_tokens_per_sec': 95},
    {'model_id': 'qwen-1.7b', 'size_mb': 1161, 'avg_tokens_per_sec': 75},
    {'model_id': 'smollm-1.7b', 'size_mb': 1161, 'avg_tokens_per_sec': 72},
    {'model_id': 'lfm2-vl-1.6b', 'size_mb': 1440, 'avg_tokens_per_sec': 60},
]

MODEL_BY_ID = {m['model_id']: m for m in CACTUS_MODELS}


def explain_metrics_calculation():
    """Show step-by-step how metrics are calculated."""

    print("=" * 80)
    print("PERFORMANCE METRICS CALCULATION")
    print("=" * 80)
    print()

    # Example routing decisions
    example_prompts = [
        ("Hi!", "gemma-270m"),
        ("What is 2+2?", "smollm-360m"),
        ("Explain photosynthesis", "gemma-270m"),
        ("Explain quantum entanglement", "lfm2-1.2b"),
        ("Write AVL tree code", "qwen-1.7b"),
    ]

    print("Example Routing Decisions:")
    print("-" * 80)
    for i, (prompt, model_id) in enumerate(example_prompts, 1):
        size = MODEL_BY_ID[model_id]['size_mb']
        tokens_sec = MODEL_BY_ID[model_id]['avg_tokens_per_sec']
        print(f"{i}. \"{prompt[:40]}\"")
        print(f"   → {model_id} ({size}MB, {tokens_sec} tok/s)")
    print()

    # Step 1: Calculate average model size
    print("STEP 1: Calculate Average Model Size")
    print("-" * 80)

    sizes = [MODEL_BY_ID[model_id]['size_mb'] for _, model_id in example_prompts]
    print(f"Selected models: {[m for _, m in example_prompts]}")
    print(f"Model sizes: {sizes} MB")
    print(f"Average = ({' + '.join(map(str, sizes))}) / {len(sizes)}")

    avg_size = sum(sizes) / len(sizes)
    print(f"Average = {avg_size:.1f} MB")
    print()

    # Step 2: Calculate savings vs always using largest
    print("STEP 2: Calculate Savings vs Always Using Largest")
    print("-" * 80)

    largest_size = 1440  # lfm2-vl-1.6b
    print(f"Largest model: lfm2-vl-1.6b = {largest_size}MB")
    print(f"Average used: {avg_size:.1f}MB")
    print(f"Savings = (Largest - Average) / Largest × 100%")
    print(f"Savings = ({largest_size} - {avg_size:.1f}) / {largest_size} × 100%")

    savings = (largest_size - avg_size) / largest_size * 100
    print(f"Savings = {savings:.1f}%")
    print()

    # Step 3: Estimate latency improvement
    print("STEP 3: Estimate Latency Improvement")
    print("-" * 80)

    print("Assumption: Latency ∝ Model Size")
    print("  - Smaller models load faster")
    print("  - Smaller models generate tokens faster")
    print("  - Latency improvement ≈ 80% of size reduction")
    print()
    print(f"Latency improvement = Savings × 0.8")
    print(f"Latency improvement = {savings:.1f}% × 0.8")

    latency_improvement = savings * 0.8
    print(f"Latency improvement ≈ {latency_improvement:.1f}%")
    print()

    # Step 4: Estimate battery savings
    print("STEP 4: Estimate Battery Savings")
    print("-" * 80)

    print("Assumption: Battery usage ∝ Model Size × Inference Time")
    print("  - Smaller models use less RAM")
    print("  - Smaller models compute faster (less CPU time)")
    print("  - Battery savings ≈ 70% of size reduction")
    print()
    print(f"Battery savings = Savings × 0.7")
    print(f"Battery savings = {savings:.1f}% × 0.7")

    battery_savings = savings * 0.7
    print(f"Battery savings ≈ {battery_savings:.1f}%")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Average model size used: {avg_size:.0f}MB")
    print(f"Savings vs always using largest: {savings:.1f}%")
    print(f"Expected latency improvement: ~{latency_improvement:.0f}%")
    print(f"Expected battery savings: ~{battery_savings:.0f}%")
    print()

    return {
        'avg_size': avg_size,
        'savings': savings,
        'latency_improvement': latency_improvement,
        'battery_savings': battery_savings,
        'sizes': sizes,
        'prompts': [p for p, _ in example_prompts]
    }


def visualize_metrics(data):
    """Create visualizations."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('AuroraAI Router Performance Metrics', fontsize=16, fontweight='bold')

    # 1. Model size distribution
    ax = axes[0, 0]
    ax.bar(range(len(data['sizes'])), data['sizes'], color='skyblue', edgecolor='navy', linewidth=1.5)
    ax.axhline(data['avg_size'], color='red', linestyle='--', linewidth=2, label=f'Average: {data["avg_size"]:.0f}MB')
    ax.axhline(1440, color='orange', linestyle='--', linewidth=2, label='Largest: 1440MB')
    ax.set_xlabel('Prompt #', fontsize=12)
    ax.set_ylabel('Model Size (MB)', fontsize=12)
    ax.set_title('Model Sizes Selected by Router', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Savings breakdown
    ax = axes[0, 1]
    categories = ['Size\nSavings', 'Latency\nImprovement', 'Battery\nSavings']
    values = [data['savings'], data['latency_improvement'], data['battery_savings']]
    colors = ['#4CAF50', '#2196F3', '#FFC107']

    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Savings / Improvement (%)', fontsize=12)
    ax.set_title('Performance Improvements', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 3. Model size comparison
    ax = axes[1, 0]
    strategies = ['Always\nSmallest\n(172MB)', 'Router\nAverage\n({:.0f}MB)'.format(data['avg_size']),
                  'Always\nLargest\n(1440MB)']
    strategy_sizes = [172, data['avg_size'], 1440]
    colors = ['green', 'blue', 'red']

    bars = ax.bar(strategies, strategy_sizes, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Average Model Size (MB)', fontsize=12)
    ax.set_title('Router vs Baseline Strategies', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 4. Calculation formula
    ax = axes[1, 1]
    ax.axis('off')

    formula_text = f"""
Metrics Calculation:

1. Average Size:
   Σ(model sizes) / n = {data['avg_size']:.1f}MB

2. Savings:
   (Largest - Avg) / Largest × 100%
   = ({1440} - {data['avg_size']:.1f}) / {1440} × 100%
   = {data['savings']:.1f}%

3. Latency Improvement:
   Savings × 0.8 = {data['latency_improvement']:.1f}%

4. Battery Savings:
   Savings × 0.7 = {data['battery_savings']:.1f}%
"""

    ax.text(0.1, 0.9, formula_text, fontsize=11, verticalalignment='top',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig('performance_metrics_visualization.png', dpi=150, bbox_inches='tight')
    print("📊 Saved visualization: performance_metrics_visualization.png")
    plt.show()


def load_benchmark_results():
    """Load and visualize benchmark results if available."""

    if not Path('benchmark_results.json').exists():
        print("⚠️  No benchmark results found. Run benchmark_router.py first.")
        return

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS VISUALIZATION")
    print("=" * 80 + "\n")

    with open('benchmark_results.json', 'r') as f:
        results = json.load(f)

    # Create comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('AuroraAI Router Benchmark Results', fontsize=16, fontweight='bold')

    strategies = list(results.keys())
    metrics_to_plot = [
        ('avg_size_mb', 'Average Model Size (MB)', axes[0, 0]),
        ('agreement_with_opus', 'Agreement with Claude Opus (%)', axes[0, 1]),
        ('savings_vs_largest', 'Savings vs Always Largest (%)', axes[1, 0]),
        ('latency_improvement', 'Latency Improvement (%)', axes[1, 1]),
    ]

    for metric_key, title, ax in metrics_to_plot:
        values = [results[s][metric_key] for s in strategies]
        colors = ['#2196F3' if s == 'aurora_router' else '#9E9E9E' for s in strategies]

        bars = ax.bar(range(len(strategies)), values, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('benchmark_comparison.png', dpi=150, bbox_inches='tight')
    print("📊 Saved benchmark visualization: benchmark_comparison.png")
    plt.show()


if __name__ == "__main__":
    # Explain calculation
    data = explain_metrics_calculation()

    # Visualize
    visualize_metrics(data)

    # Load benchmark results if available
    load_benchmark_results()
