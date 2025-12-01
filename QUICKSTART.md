# 🚀 Quick Start Guide - AuroraAI Router

## Step-by-Step Testing Instructions

Follow these steps to test the router right now!

---

## ✅ Step 1: Install Dependencies

```bash
cd c:\Users\House Computer\Downloads\auroraai\auroraai-router

# Install Python packages
pip install -r requirements.txt
```

---

## ✅ Step 2: Create a Router Profile

Run the profiling notebook to create a router profile:

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/01_profile_cactus_models.ipynb
# Run all cells (Cell -> Run All)
```

**What this does:**
- Creates synthetic test dataset (500 prompts)
- Extracts embeddings using SentenceTransformers
- Clusters prompts into 10 groups
- Simulates model performance (small models = higher error, large = lower)
- Saves profile to `profiles/cactus_models_profile.json`

**Output:**
- `profiles/cactus_models_profile.json` (2-3MB)
- `profiles/clusters_visualization.png`
- `profiles/error_rates_heatmap.png`

---

## ✅ Step 3: Test the Router

### Option A: Use Jupyter Notebook

```bash
# Open: notebooks/02_test_routing.ipynb
# Run all cells
```

This will:
- Load the router profile
- Test routing with different prompts
- Show which model is selected for each prompt
- Benchmark routing speed (~15ms per request)

### Option B: Run Python Example

```bash
python examples/python/example_basic.py
```

**Expected Output:**
```
AuroraAI Router initialized
Router info: {'n_clusters': 10, 'n_models': 5, ...}

Testing routing decisions:
--------------------------------------------------------------------------------

Simple greeting
Prompt: 'Hi, how are you?'
Cost preference: 0.2 (0=fast, 1=quality)

✓ Selected Model: gemma-270m
  Path: weights/gemma-3-270m-it
  Score: 0.187
  Cluster: 3
  Est. latency: 289ms
  Alternatives:
    - smollm-360m (score: 0.203)
    - qwen-600m (score: 0.245)
--------------------------------------------------------------------------------

Complex explanation
Prompt: 'Explain how neural networks work'
Cost preference: 0.7 (0=fast, 1=quality)

✓ Selected Model: qwen-1.7b
  Path: weights/Qwen3-1.7B
  Score: 0.089
  Cluster: 7
  Est. latency: 667ms
--------------------------------------------------------------------------------
```

---

## ✅ Step 4: Run Unit Tests

Verify everything works:

```bash
python tests/test_mobile_router.py
```

**Expected Output:**
```
test_assign_cluster (__main__.TestMobileClusterEngine) ... ok
test_initialization (__main__.TestMobileClusterEngine) ... ok
test_save_load (__main__.TestMobileClusterEngine) ... ok
test_available_models_filter (__main__.TestMobileRouter) ... ok
test_get_cluster_info (__main__.TestMobileRouter) ... ok
test_get_supported_models (__main__.TestMobileRouter) ... ok
test_route_to_large_model (__main__.TestMobileRouter) ... ok
test_route_to_small_model (__main__.TestMobileRouter) ... ok
test_create_cactus_profile (__main__.TestProfileConverter) ... ok

----------------------------------------------------------------------
Ran 9 tests in 1.234s

OK
```

---

## ✅ Step 5: Test with Your Own Prompts

Create a simple test script:

```python
# test_my_prompts.py
from sdks.python.auroraai_router import AuroraAIRouter, ModelInfo

# Define models
models = [
    ModelInfo(
        model_id='gemma-270m',
        model_path='weights/gemma-270m',
        size_mb=172,
        avg_tokens_per_sec=173
    ),
    ModelInfo(
        model_id='qwen-1.7b',
        model_path='weights/qwen-1.7b',
        size_mb=1161,
        avg_tokens_per_sec=75
    ),
]

# Load router
router = AuroraAIRouter(
    profile_path='profiles/cactus_models_profile.json',
    models=models
)

# Test your prompts!
my_prompts = [
    "Hello!",
    "Write a Python function to reverse a string",
    "Explain quantum entanglement in detail",
]

for prompt in my_prompts:
    # Try different cost preferences
    for cost_pref in [0.3, 0.5, 0.8]:
        result = router.route(prompt, cost_preference=cost_pref)
        print(f"{prompt[:30]:30s} | cost={cost_pref:.1f} | {result.model_id:15s}")
    print()
```

Run it:
```bash
python test_my_prompts.py
```

---

## 🎯 Understanding the Results

### What Cost Preference Means:

- **0.0 - 0.3** = "I want fast responses, don't care about quality"
  - Routes to smallest models (Gemma-270m, SmolLM-360m)
  - Best for: Simple greetings, basic Q&A

- **0.4 - 0.6** = "Balance speed and quality"
  - Routes to medium models (Qwen-600m, LFM2-700M)
  - Best for: General tasks

- **0.7 - 1.0** = "I want best quality, don't care about speed"
  - Routes to largest models (Qwen-1.7B)
  - Best for: Complex reasoning, coding, detailed explanations

### Routing Score Explained:

```
score = error_rate[cluster] + λ × normalized_size

Where:
- error_rate[cluster] = how often this model fails on prompts in this cluster
- λ (lambda) = cost penalty (calculated from cost_preference)
- normalized_size = model size normalized to [0, 1]

Lower score = better choice!
```

---

## 🔧 Troubleshooting

### Error: "sentence-transformers not installed"

```bash
pip install sentence-transformers
```

### Error: "Profile not found"

Make sure you ran notebook `01_profile_cactus_models.ipynb` first to create the profile.

### Error: "CUDA not available" (warning, not error)

This is OK! The router will use CPU, which is fine for mobile simulation.

### Slow embedding extraction?

First run is slower (downloads model). Subsequent runs are much faster.

---

## 🎓 Next Steps

### 1. Create Real Model Profiles

Replace the simulated performance in `01_profile_cactus_models.ipynb` with actual Cactus inference:

```python
# Instead of simulating:
# error_rate = simulate_model_performance(...)

# Use real Cactus inference:
import cactus  # Your Cactus bindings

model = cactus.init(model_path, 2048)
for prompt, expected in dataset:
    response = model.complete([{"role": "user", "content": prompt}])
    actual_output = response['response']
    is_correct = (actual_output.strip().lower() == expected.strip().lower())
    # Track correctness per cluster...
```

### 2. Test on Real Devices

- Copy `auroraai-router/` folder to your Android/iOS device
- Run the Python code on-device (using Termux on Android or similar)
- Measure actual routing latency

### 3. Integrate with Cactus

```python
from auroraai_router import AuroraAIRouter
import cactus  # Your Cactus library

router = AuroraAIRouter('profile.json', models)

# User sends a message
user_prompt = "Explain AI"

# Route to optimal model
result = router.route(user_prompt, cost_preference=0.6)

# Load and run with Cactus
model = cactus.init(result.model_path, 2048)
response = model.complete([
    {"role": "user", "content": user_prompt}
])

print(f"Model: {result.model_id}")
print(f"Response: {response}")
```

### 4. Build Android/iOS App

Use the C++ router (in `router-native/`) or Python router with mobile Python runtime:

- **Android**: Use Chaquopy (Python in Android) or compile C++ router as JNI library
- **iOS**: Use PythonKit or compile C++ router as framework
- **Flutter**: Create platform channel to call router

---

## ✨ You're All Set!

The router is now fully functional. Key files:

- ✅ **Core library**: `core/mobile_router.py`
- ✅ **Profile**: `profiles/cactus_models_profile.json`
- ✅ **Examples**: `examples/python/example_basic.py`
- ✅ **Tests**: `tests/test_mobile_router.py`
- ✅ **Notebooks**: `notebooks/*.ipynb`

**Happy routing! 🚀**
