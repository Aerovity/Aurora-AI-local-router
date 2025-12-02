#!/usr/bin/env python3
"""
Lambda Parameter Analysis for AuroraAI Router
==============================================

Creates 2 graphs showing how lambda (cost_preference) affects:
1. Quality (Task Success Rate) vs Lambda
2. Cost (Model Size) vs Lambda

Lambda controls the quality-cost tradeoff:
- Lambda = 0.0 → Prefer smallest/fastest models (low cost, lower quality)
- Lambda = 1.0 → Prefer largest/best models (high cost, higher quality)
"""

import sys
import os
import time
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sdks', 'python'))

import warnings
warnings.filterwarnings('ignore')

print("🔍 Loading dependencies...")

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("❌ matplotlib and numpy required: pip install matplotlib numpy")
    sys.exit(1)

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

from auroraai_router import AuroraAIRouter

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CACTUS_MODELS = [
    {'model_id': 'gemma-270m', 'model_path': 'google/gemma-3-270m-it', 'size_mb': 172, 'avg_tokens_per_sec': 173},
    {'model_id': 'smollm-360m', 'model_path': 'HuggingFaceTB/SmolLM2-360m-Instruct', 'size_mb': 227, 'avg_tokens_per_sec': 150},
    {'model_id': 'lfm2-350m', 'model_path': 'LiquidAI/LFM2-350M', 'size_mb': 233, 'avg_tokens_per_sec': 145},
    {'model_id': 'qwen-600m', 'model_path': 'Qwen/Qwen3-0.6B', 'size_mb': 394, 'avg_tokens_per_sec': 129},
    {'model_id': 'lfm2-vl-450m', 'model_path': 'LiquidAI/LFM2-VL-450M', 'size_mb': 420, 'avg_tokens_per_sec': 113},
    {'model_id': 'lfm2-700m', 'model_path': 'LiquidAI/LFM2-700M', 'size_mb': 467, 'avg_tokens_per_sec': 115},
    {'model_id': 'gemma-1b', 'model_path': 'google/gemma-3-1b-it', 'size_mb': 642, 'avg_tokens_per_sec': 100},
    {'model_id': 'lfm2-1.2b', 'model_path': 'LiquidAI/LFM2-1.2B', 'size_mb': 722, 'avg_tokens_per_sec': 95},
    {'model_id': 'lfm2-1.2b-tools', 'model_path': 'LiquidAI/LFM2-1.2B-Tools', 'size_mb': 722, 'avg_tokens_per_sec': 95},
    {'model_id': 'qwen-1.7b', 'model_path': 'Qwen/Qwen3-1.7B', 'size_mb': 1161, 'avg_tokens_per_sec': 75},
    {'model_id': 'smollm-1.7b', 'model_path': 'HuggingFaceTB/SmolLM2-1.7B-Instruct', 'size_mb': 1161, 'avg_tokens_per_sec': 72},
    {'model_id': 'lfm2-vl-1.6b', 'model_path': 'LiquidAI/LFM2-VL-1.6B', 'size_mb': 1440, 'avg_tokens_per_sec': 60},
]

MODEL_TIERS = {
    'gemma-270m': 1, 'smollm-360m': 1, 'lfm2-350m': 1,
    'qwen-600m': 2, 'lfm2-vl-450m': 2, 'lfm2-700m': 2,
    'gemma-1b': 3, 'lfm2-1.2b': 3, 'lfm2-1.2b-tools': 3,
    'qwen-1.7b': 4, 'smollm-1.7b': 4,
    'lfm2-vl-1.6b': 5,
}

MODEL_BY_ID = {m['model_id']: m for m in CACTUS_MODELS}

SUBJECT_COMPLEXITY = {
    "high_school_geography": 1, "marketing": 1, "nutrition": 1, "sociology": 1,
    "anatomy": 2, "astronomy": 2, "business_ethics": 2, "computer_security": 2,
    "college_biology": 3, "college_chemistry": 3, "philosophy": 3, "virology": 3,
    "econometrics": 4, "electrical_engineering": 4, "formal_logic": 4,
    "abstract_algebra": 5, "college_mathematics": 5, "college_physics": 5,
}


def get_complexity(subject: str) -> int:
    return SUBJECT_COMPLEXITY.get(subject, 3)


def model_can_handle(model_id: str, complexity: int) -> bool:
    return MODEL_TIERS.get(model_id, 0) >= complexity


def load_samples(n_samples: int = 100) -> List[Dict]:
    """Load MMLU samples."""
    if not HAS_DATASETS:
        print("  📝 Using synthetic samples...")
        samples = []
        prompts = {
            1: ["What is the capital of France?", "Define photosynthesis"],
            2: ["Explain how vaccines work", "What causes earthquakes?"],
            3: ["Compare mitosis and meiosis", "Explain quantum entanglement"],
            4: ["Derive the quadratic formula", "Explain P vs NP"],
            5: ["Explain Gödel's theorems", "What is Riemann hypothesis?"],
        }
        per_tier = n_samples // 5
        for tier in range(1, 6):
            for i in range(per_tier):
                samples.append({
                    'question': prompts[tier][i % len(prompts[tier])],
                    'complexity': tier
                })
        random.shuffle(samples)
        return samples
    
    print("  📚 Loading MMLU...")
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    
    samples_by_complexity = defaultdict(list)
    for sample in mmlu:
        complexity = get_complexity(sample['subject'])
        samples_by_complexity[complexity].append({
            'question': sample['question'],
            'complexity': complexity
        })
    
    final = []
    per_level = n_samples // 5
    for c in range(1, 6):
        available = samples_by_complexity[c]
        random.shuffle(available)
        final.extend(available[:per_level])
    
    random.shuffle(final)
    return final[:n_samples]


def evaluate_lambda(router, samples: List[Dict], lambda_value: float) -> Dict:
    """Evaluate router at a specific lambda (cost_preference) value."""
    successes = 0
    total_size = 0
    
    for sample in samples:
        try:
            result = router.route(sample['question'], cost_preference=lambda_value)
            model_id = result.model_id
        except:
            model_id = 'qwen-1.7b'
        
        if model_can_handle(model_id, sample['complexity']):
            successes += 1
        
        total_size += MODEL_BY_ID.get(model_id, {}).get('size_mb', 722)
    
    return {
        'quality': (successes / len(samples)) * 100,
        'avg_cost': total_size / len(samples)
    }


def create_lambda_graphs():
    """Create Quality vs Lambda and Cost vs Lambda graphs."""
    
    print()
    print("╔" + "═" * 60 + "╗")
    print("║" + " 📊 LAMBDA PARAMETER ANALYSIS ".center(60) + "║")
    print("╚" + "═" * 60 + "╝")
    print()
    
    # Initialize router
    print("🔧 Initializing router...")
    router = AuroraAIRouter(
        profile_path='profiles/production/cactus_production_profile.json',
        models=CACTUS_MODELS,
        embedding_model_name='BAAI/bge-base-en-v1.5'
    )
    print("  ✅ Router loaded")
    
    # Load samples
    samples = load_samples(100)
    print(f"  ✅ Loaded {len(samples)} samples")
    print()
    
    # Lambda values to test
    lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    print("⏱️  Evaluating lambda values...")
    results = []
    
    for lam in lambda_values:
        print(f"  Testing λ = {lam:.1f}...")
        metrics = evaluate_lambda(router, samples, lam)
        results.append({
            'lambda': lam,
            'quality': metrics['quality'],
            'cost': metrics['avg_cost']
        })
    
    print()
    
    # Extract data for plotting
    lambdas = [r['lambda'] for r in results]
    qualities = [r['quality'] for r in results]
    costs = [r['cost'] for r in results]
    
    # Print results table
    print("╔" + "═" * 50 + "╗")
    print("║" + " RESULTS ".center(50) + "║")
    print("╠" + "═" * 50 + "╣")
    print("║  " + f"{'Lambda':<10} {'Quality (%)':<15} {'Avg Cost (MB)':<15}" + "   ║")
    print("╟" + "─" * 50 + "╢")
    for r in results:
        print(f"║  {r['lambda']:<10.1f} {r['quality']:<15.1f} {r['cost']:<15.0f}   ║")
    print("╚" + "═" * 50 + "╝")
    print()
    
    # ══════════════════════════════════════════════════════════════════════════
    # GRAPH 1: Quality vs Lambda (separate file)
    # ══════════════════════════════════════════════════════════════════════════
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(lambdas, qualities, 'o-', color='#27AE60', linewidth=3, markersize=12, 
             markerfacecolor='white', markeredgewidth=3, label='Quality')
    ax1.fill_between(lambdas, qualities, alpha=0.2, color='#27AE60')
    
    ax1.set_xlabel('Lambda (cost_preference)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Quality (%)', fontsize=14, fontweight='bold')
    ax1.set_title('📊 Quality vs Lambda\nAuroraAI Router - MMLU Benchmark', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(0, 110)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add data point labels
    for x, y in zip(lambdas, qualities):
        ax1.annotate(f'{y:.0f}%', xy=(x, y), xytext=(0, 10), 
                    textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
    
    # Add annotations
    ax1.annotate('Prefer speed\n(smallest models)', xy=(0.1, 25), fontsize=11, color='#666', ha='center')
    ax1.annotate('Prefer quality\n(largest models)', xy=(0.9, 90), fontsize=11, color='#666', ha='center')
    
    plt.tight_layout()
    plt.savefig('quality_vs_lambda.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig('quality_vs_lambda.pdf', bbox_inches='tight')
    plt.close(fig1)
    
    print("📈 Saved quality_vs_lambda.png/pdf")
    
    # ══════════════════════════════════════════════════════════════════════════
    # GRAPH 2: Cost vs Lambda (separate file)
    # ══════════════════════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    ax2.plot(lambdas, costs, 's-', color='#E74C3C', linewidth=3, markersize=12,
             markerfacecolor='white', markeredgewidth=3, label='Cost')
    ax2.fill_between(lambdas, costs, alpha=0.2, color='#E74C3C')
    
    ax2.set_xlabel('Lambda (cost_preference)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Cost (MB)', fontsize=14, fontweight='bold')
    ax2.set_title('💰 Cost vs Lambda\nAuroraAI Router - MMLU Benchmark', fontsize=16, fontweight='bold', pad=15)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(0, 1600)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add reference lines
    ax2.axhline(y=172, color='#3498DB', linestyle='--', linewidth=2, alpha=0.7, label='Smallest model (172MB)')
    ax2.axhline(y=1440, color='#9B59B6', linestyle='--', linewidth=2, alpha=0.7, label='Largest model (1440MB)')
    ax2.legend(loc='upper left', fontsize=10)
    
    # Add data point labels
    for x, y in zip(lambdas, costs):
        ax2.annotate(f'{y:.0f}', xy=(x, y), xytext=(0, 10), 
                    textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
    
    # Add annotations
    ax2.annotate('Low cost', xy=(0.1, 250), fontsize=11, color='#666', ha='center')
    ax2.annotate('High cost', xy=(0.9, 1350), fontsize=11, color='#666', ha='center')
    
    plt.tight_layout()
    plt.savefig('cost_vs_lambda.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig('cost_vs_lambda.pdf', bbox_inches='tight')
    plt.close(fig2)
    
    print("📈 Saved cost_vs_lambda.png/pdf")
    print()
    print("╔" + "═" * 60 + "╗")
    print("║" + " ✅ ANALYSIS COMPLETE ".center(60) + "║")
    print("╚" + "═" * 60 + "╝")


if __name__ == "__main__":
    create_lambda_graphs()
