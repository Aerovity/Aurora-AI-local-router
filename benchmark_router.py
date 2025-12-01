#!/usr/bin/env python3
"""
Benchmark AuroraAI Router against:
1. Claude Opus 4.5 (as routing oracle - 100% accuracy baseline)
2. Gemini Pro (for routing decisions)
3. Local LLMs (Qwen, Gemma, etc. for routing)
4. Always using largest model (baseline)
5. Always using smallest model (baseline)
"""

import sys
import os
import json
import time
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sdks', 'python'))

# Optional imports
try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("⚠️  anthropic not installed. Install with: pip install anthropic")

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("⚠️  google-generativeai not installed. Install with: pip install google-generativeai")

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("⚠️  datasets not installed. Install with: pip install datasets")

from auroraai_router import AuroraAIRouter

# API Keys (will be set from command line or environment)
ANTHROPIC_API_KEY = None
GEMINI_API_KEY = None

# Cactus models
CACTUS_MODELS = [
    {'model_id': 'gemma-270m', 'model_path': 'google/gemma-3-270m-it', 'size_mb': 172, 'avg_tokens_per_sec': 173},
    {'model_id': 'lfm2-350m', 'model_path': 'LiquidAI/LFM2-350M', 'size_mb': 233, 'avg_tokens_per_sec': 145},
    {'model_id': 'smollm-360m', 'model_path': 'HuggingFaceTB/SmolLM2-360m-Instruct', 'size_mb': 227, 'avg_tokens_per_sec': 150},
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

MODEL_BY_ID = {m['model_id']: m for m in CACTUS_MODELS}


def load_mmlu_samples(n_samples: int = 200) -> List[Dict]:
    """Load MMLU test samples."""
    if not HAS_DATASETS:
        print("❌ datasets library not installed")
        return []

    print(f"📚 Loading {n_samples} MMLU samples...")
    mmlu = load_dataset("cais/mmlu", "all", split="test")

    # Diverse topics
    topics = [
        "abstract_algebra", "anatomy", "computer_security", "astronomy",
        "international_law", "marketing", "philosophy", "electrical_engineering",
        "econometrics", "moral_scenarios", "professional_medicine", "virology"
    ]

    samples = []
    per_topic = n_samples // len(topics)

    for topic in topics:
        topic_samples = [s for s in mmlu if s["subject"] == topic]
        samples.extend(topic_samples[:per_topic])

    return samples[:n_samples]


def route_with_claude_opus(prompt: str, client: Anthropic) -> str:
    """Use Claude Opus 4.5 to select best model (ground truth)."""
    model_list = "\n".join([f"- {m['model_id']} ({m['size_mb']}MB)" for m in CACTUS_MODELS])

    system_prompt = f"""You are a routing expert. Given a user prompt, select the SMALLEST model that can handle it well.

Available models (smallest to largest):
{model_list}

Rules:
- Use smallest models (gemma-270m, smollm-360m) for: greetings, simple Q&A, basic math
- Use medium models (qwen-600m, lfm2-700m, gemma-1b) for: general knowledge, simple explanations
- Use large models (lfm2-1.2b, qwen-1.7b) for: complex reasoning, detailed explanations
- Use largest model (lfm2-vl-1.6b) for: very complex tasks, coding, analysis

Respond with ONLY the model_id, nothing else."""

    try:
        response = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=50,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": f"Select best model for: {prompt}"}]
        )

        selected = response.content[0].text.strip()
        # Extract model_id if it's in the response
        for m in CACTUS_MODELS:
            if m['model_id'] in selected:
                return m['model_id']

        return selected
    except Exception as e:
        print(f"  ⚠️  Claude error: {e}")
        return None


def route_with_gemini(prompt: str, model) -> str:
    """Use Gemini to select best model."""
    model_list = "\n".join([f"- {m['model_id']} ({m['size_mb']}MB)" for m in CACTUS_MODELS])

    system_prompt = f"""Select the SMALLEST model that can handle this task well.

Available models:
{model_list}

Respond with ONLY the model_id."""

    try:
        response = model.generate_content(f"{system_prompt}\n\nTask: {prompt}")
        selected = response.text.strip()

        for m in CACTUS_MODELS:
            if m['model_id'] in selected:
                return m['model_id']

        return selected
    except Exception as e:
        print(f"  ⚠️  Gemini error: {e}")
        return None


def calculate_metrics(selections: Dict[str, List[str]], ground_truth: List[str]) -> Dict:
    """Calculate performance metrics for each routing strategy."""
    metrics = {}

    for strategy, selected_models in selections.items():
        # Calculate sizes
        sizes = []
        for model_id in selected_models:
            if model_id and model_id in MODEL_BY_ID:
                sizes.append(MODEL_BY_ID[model_id]['size_mb'])
            else:
                sizes.append(722)  # Default to medium model

        avg_size = sum(sizes) / len(sizes) if sizes else 0

        # Agreement with ground truth
        agreement = sum(1 for i, m in enumerate(selected_models) if m == ground_truth[i]) / len(ground_truth) * 100

        # Calculate savings
        largest_size = 1440  # lfm2-vl-1.6b
        savings = (largest_size - avg_size) / largest_size * 100

        # Estimate latency improvement (proportional to size reduction)
        latency_improvement = savings * 0.8  # 80% of size savings
        battery_savings = savings * 0.7  # 70% of size savings

        metrics[strategy] = {
            'avg_size_mb': avg_size,
            'min_size_mb': min(sizes) if sizes else 0,
            'max_size_mb': max(sizes) if sizes else 0,
            'agreement_with_opus': agreement,
            'savings_vs_largest': savings,
            'latency_improvement': latency_improvement,
            'battery_savings': battery_savings,
            'selections': selected_models
        }

    return metrics


def run_benchmark(n_samples: int = 100):
    """Run comprehensive benchmark."""

    print("=" * 80)
    print("AURORA AI ROUTER BENCHMARK")
    print("=" * 80)
    print()

    # Check dependencies
    if not HAS_ANTHROPIC or not ANTHROPIC_API_KEY:
        print("❌ Claude Opus not available (need anthropic library + API key)")
        return

    if not HAS_DATASETS:
        print("❌ datasets library not available")
        return

    # Initialize
    print("🔧 Initializing...")

    # Load router
    router = AuroraAIRouter(
        profile_path='profiles/production/cactus_production_profile.json',
        models=CACTUS_MODELS,
        embedding_model_name='BAAI/bge-base-en-v1.5'
    )
    print("  ✅ AuroraAI Router loaded")

    # Initialize Claude
    claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    print("  ✅ Claude Opus 4.5 initialized")

    # Initialize Gemini (optional)
    gemini_model = None
    if HAS_GEMINI and GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-pro')
        print("  ✅ Gemini Pro initialized")

    # Load test data
    samples = load_mmlu_samples(n_samples)
    print(f"  ✅ Loaded {len(samples)} MMLU samples")
    print()

    # Run benchmark
    print("=" * 80)
    print("RUNNING BENCHMARK")
    print("=" * 80)
    print()

    selections = {
        'aurora_router': [],
        'claude_opus': [],
        'gemini': [] if gemini_model else None,
        'always_largest': [],
        'always_smallest': [],
    }

    for i, sample in enumerate(samples, 1):
        prompt = sample['question']

        if i % 20 == 0:
            print(f"Progress: {i}/{len(samples)} samples...")

        # 1. AuroraAI Router
        try:
            result = router.route(prompt, cost_preference=0.7)
            selections['aurora_router'].append(result.model_id)
        except Exception as e:
            print(f"  ⚠️  Router error: {e}")
            selections['aurora_router'].append('qwen-1.7b')  # Default

        # 2. Claude Opus (ground truth)
        opus_choice = route_with_claude_opus(prompt, claude_client)
        selections['claude_opus'].append(opus_choice if opus_choice else 'qwen-1.7b')

        # 3. Gemini (if available)
        if gemini_model:
            gemini_choice = route_with_gemini(prompt, gemini_model)
            selections['gemini'].append(gemini_choice if gemini_choice else 'qwen-1.7b')

        # 4. Baselines
        selections['always_largest'].append('lfm2-vl-1.6b')
        selections['always_smallest'].append('gemma-270m')

        # Rate limiting
        time.sleep(0.1)

    print()
    print("=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print()

    # Calculate metrics
    if selections['gemini'] is None:
        del selections['gemini']

    metrics = calculate_metrics(selections, selections['claude_opus'])

    # Print results
    print(f"{'Strategy':<20} {'Avg Size':<12} {'Agreement':<12} {'Savings':<12} {'Latency↓':<12} {'Battery↓'}")
    print("-" * 80)

    for strategy, data in metrics.items():
        print(f"{strategy:<20} {data['avg_size_mb']:>8.0f}MB   "
              f"{data['agreement_with_opus']:>8.1f}%   "
              f"{data['savings_vs_largest']:>8.1f}%   "
              f"{data['latency_improvement']:>8.1f}%   "
              f"{data['battery_savings']:>8.1f}%")

    print()

    # Save results
    output_file = 'benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Results saved to {output_file}")
    print()

    # Print insights
    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()

    aurora_metrics = metrics['aurora_router']
    opus_metrics = metrics['claude_opus']

    print(f"🎯 AuroraAI Router Performance:")
    print(f"   - Agreement with Claude Opus: {aurora_metrics['agreement_with_opus']:.1f}%")
    print(f"   - Average model size: {aurora_metrics['avg_size_mb']:.0f}MB")
    print(f"   - Savings vs always largest: {aurora_metrics['savings_vs_largest']:.1f}%")
    print(f"   - Expected latency improvement: ~{aurora_metrics['latency_improvement']:.0f}%")
    print(f"   - Expected battery savings: ~{aurora_metrics['battery_savings']:.0f}%")
    print()

    print(f"🏆 Claude Opus (Ground Truth):")
    print(f"   - Average model size: {opus_metrics['avg_size_mb']:.0f}MB")
    print(f"   - Savings vs always largest: {opus_metrics['savings_vs_largest']:.1f}%")
    print()

    if 'gemini' in metrics:
        gemini_metrics = metrics['gemini']
        print(f"💎 Gemini Pro:")
        print(f"   - Agreement with Claude Opus: {gemini_metrics['agreement_with_opus']:.1f}%")
        print(f"   - Average model size: {gemini_metrics['avg_size_mb']:.0f}MB")
        print()

    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark AuroraAI Router")
    parser.add_argument('--anthropic-key', type=str, help='Anthropic API key')
    parser.add_argument('--gemini-key', type=str, help='Gemini API key')
    parser.add_argument('--n-samples', type=int, default=100, help='Number of samples to test')

    args = parser.parse_args()

    ANTHROPIC_API_KEY = args.anthropic_key or os.getenv('ANTHROPIC_API_KEY')
    GEMINI_API_KEY = args.gemini_key or os.getenv('GEMINI_API_KEY')

    run_benchmark(args.n_samples)
