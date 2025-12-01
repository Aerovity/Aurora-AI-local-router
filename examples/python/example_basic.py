"""Basic example of using AuroraAI Router with Cactus models."""

import sys
from pathlib import Path

# Add SDK to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'sdks' / 'python'))

from auroraai_router import AuroraAIRouter, ModelInfo

def main():
    """Run basic routing example."""

    # Define Cactus models available on your device
    cactus_models = [
        ModelInfo(
            model_id='gemma-270m',
            model_path='weights/gemma-3-270m-it',
            size_mb=172,
            avg_tokens_per_sec=173,
            capabilities=['text']
        ),
        ModelInfo(
            model_id='qwen-600m',
            model_path='weights/Qwen3-0.6B',
            size_mb=394,
            avg_tokens_per_sec=129,
            capabilities=['text', 'tools']
        ),
        ModelInfo(
            model_id='qwen-1.7b',
            model_path='weights/Qwen3-1.7B',
            size_mb=1161,
            avg_tokens_per_sec=75,
            capabilities=['text', 'tools']
        ),
    ]

    # Initialize router
    profile_path = Path(__file__).parent.parent.parent / 'profiles' / 'cactus_models_profile.json'
    router = AuroraAIRouter(
        profile_path=str(profile_path),
        models=cactus_models
    )

    print("AuroraAI Router initialized")
    print(f"Router info: {router.get_info()}\n")

    # Test different prompts
    test_cases = [
        ("Hi, how are you?", 0.2, "Simple greeting"),
        ("What is 2+2?", 0.3, "Simple math"),
        ("Explain how neural networks work", 0.7, "Complex explanation"),
        ("Write Python code for quicksort", 0.5, "Coding task"),
    ]

    print("Testing routing decisions:\n")
    print("-" * 80)

    for prompt, cost_pref, description in test_cases:
        result = router.route(
            prompt=prompt,
            cost_preference=cost_pref,
            return_alternatives=True
        )

        print(f"\n{description}")
        print(f"Prompt: '{prompt}'")
        print(f"Cost preference: {cost_pref:.1f} (0=fast, 1=quality)")
        print(f"\n✓ Selected Model: {result.model_id}")
        print(f"  Path: {result.model_path}")
        print(f"  Score: {result.score:.3f}")
        print(f"  Cluster: {result.cluster_id}")
        print(f"  Est. latency: {result.estimated_latency_ms:.0f}ms")

        if result.alternatives:
            print(f"  Alternatives:")
            for alt_id, alt_score in result.alternatives[:2]:
                print(f"    - {alt_id} (score: {alt_score:.3f})")

        print("-" * 80)

    print("\n✅ Example complete!")

    # Now you would use result.model_path with Cactus:
    # model = cactus_init(result.model_path, 2048)
    # response = cactus_complete(model, messages, ...)

if __name__ == "__main__":
    main()
