"""
Verification script to check if everything is set up correctly.

Run this to verify your setup before generating profiles.
"""

import sys
import platform
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

def print_section(title):
    """Print section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check_architecture():
    """Check system architecture."""
    print_section("System Architecture")

    machine = platform.machine().lower()
    system = platform.system()

    print(f"OS: {system}")
    print(f"Architecture: {machine}")

    if machine in ['arm64', 'aarch64']:
        print("[OK] ARM architecture detected - Cactus can run")
        return True
    else:
        print("[WARN] x86 architecture - Use --mock-embeddings mode")
        return False


def check_python_dependencies():
    """Check if required Python packages are installed."""
    print_section("Python Dependencies")

    required = [
        'datasets',
        'numpy',
        'pandas',
        'sklearn',
        'hdbscan',
        'tqdm',
    ]

    optional = [
        'sentence_transformers',  # For mock mode
    ]

    all_good = True

    print("Required packages:")
    for package in required:
        try:
            __import__(package)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [FAIL] {package} (missing)")
            all_good = False

    print("\nOptional packages:")
    for package in optional:
        try:
            __import__(package)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [WARN] {package} (not installed - needed for --mock-embeddings)")

    return all_good


def check_cactus_library():
    """Check if Cactus library is available."""
    print_section("Cactus Library")

    try:
        from bindings import is_cactus_available

        available, message = is_cactus_available()

        if available:
            print(f"[OK] Cactus library found")
            print(f"   {message}")
            return True
        else:
            print(f"[FAIL] Cactus library not found")
            print(f"   {message}")
            print("\n   To build:")
            print("   cd ../../cactus")
            print("   ./apple/build.sh")
            return False

    except Exception as e:
        print(f"[FAIL] Error checking Cactus library: {e}")
        return False


def check_models():
    """Check if models are available."""
    print_section("Embedding Models")

    cactus_dir = Path(__file__).parent.parent.parent / "cactus"
    weights_dir = cactus_dir / "weights"

    if not weights_dir.exists():
        print(f"[WARN] Weights directory not found: {weights_dir}")
        print("   No models downloaded yet")
        return False

    # Check for embedding models
    embedding_models = [
        'LFM2-350M',
        'Qwen3-0.6B',
        'LFM2-700M',
    ]

    found = []
    for model_name in embedding_models:
        model_dir = weights_dir / model_name
        if model_dir.exists():
            # Check for model.gguf
            model_file = model_dir / "model.gguf"
            if model_file.exists():
                size_mb = model_file.stat().st_size / (1024 * 1024)
                print(f"[OK] {model_name} ({size_mb:.0f}MB)")
                found.append(model_name)
            else:
                print(f"[WARN] {model_name} directory exists but model.gguf not found")

    if not found:
        print("[FAIL] No embedding models found")
        print("\n   To download:")
        print(f"   cd {cactus_dir}")
        print("   ./cli/cactus download LiquidAI/LFM2-350M")
        return False

    return True


def check_project_structure():
    """Check if all required files exist."""
    print_section("Project Structure")

    required_files = [
        'bindings/__init__.py',
        'bindings/cactus_bindings.py',
        'bindings/test_bindings.py',
        'training/__init__.py',
        'training/config.py',
        'training/utils.py',
        'training/generate_profile.py',
        'requirements.txt',
        'README.md',
    ]

    all_good = True
    base_path = Path(__file__).parent

    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"  [OK] {file_path}")
        else:
            print(f"  [FAIL] {file_path} (missing)")
            all_good = False

    return all_good


def print_recommendations(is_arm, has_deps, has_cactus, has_models):
    """Print recommendations based on checks."""
    print_section("Recommendations")

    if is_arm and has_deps and has_cactus and has_models:
        print("SUCCESS: Everything is ready!")
        print("\nYou can generate production profiles:")
        print("\n  python training/generate_profile.py \\")
        print("    --use-cactus \\")
        print("    --model-path ../../cactus/weights/LFM2-350M/model.gguf \\")
        print("    --output profiles/production_profile.json")

    elif is_arm and has_deps and has_cactus and not has_models:
        print("WARNING: Almost ready! Just need to download models:")
        print("\n  cd ../../cactus")
        print("  ./cli/cactus download LiquidAI/LFM2-350M")

    elif is_arm and has_deps and not has_cactus:
        print("WARNING: Need to build Cactus library:")
        print("\n  cd ../../cactus")
        print("  ./apple/build.sh")

    elif not is_arm and has_deps:
        print("INFO: You're on x86 - can use mock mode for testing:")
        print("\n  python training/generate_profile.py \\")
        print("    --mock-embeddings \\")
        print("    --output profiles/test_profile.json")
        print("\n  Transfer to Mac for production profiles.")

    elif not has_deps:
        print("WARNING: Install Python dependencies first:")
        print("\n  pip install -r requirements.txt")

    else:
        print("WARNING: Check the errors above and fix them")


def main():
    """Run all checks."""
    print("\n" + "=" * 60)
    print("  Cactus Profile Generation - Setup Verification")
    print("=" * 60)

    # Run checks
    is_arm = check_architecture()
    has_deps = check_python_dependencies()
    has_structure = check_project_structure()
    has_cactus = check_cactus_library() if is_arm else False
    has_models = check_models() if is_arm else False

    # Print recommendations
    print_recommendations(is_arm, has_deps, has_cactus, has_models)

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Architecture: {'[OK] ARM' if is_arm else '[WARN] x86 (mock mode only)'}")
    print(f"  Dependencies: {'[OK]' if has_deps else '[FAIL]'}")
    print(f"  Project Structure: {'[OK]' if has_structure else '[FAIL]'}")

    if is_arm:
        print(f"  Cactus Library: {'[OK]' if has_cactus else '[FAIL]'}")
        print(f"  Models: {'[OK]' if has_models else '[FAIL]'}")

    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
