# Embedding Consistency Validation Guide

## Why This Matters

Your router profile contains **cluster centers** computed from embeddings. For correct routing:

```
Training embeddings → Cluster centers → Router Profile
                                             ↓
Runtime embeddings → Cluster assignment → Model selection
       ↑                                        ↑
   MUST MATCH! ←──────────────────────────────┘
```

**If embeddings don't match → Clusters assigned incorrectly → Wrong models selected**

---

## Three Validation Levels

### Level 1: Quick Validation (No Cactus Required) ✅

**What it checks:**
- HuggingFace Nomic embeddings are deterministic
- Profile structure is valid
- Cluster assignments work
- Shows expected routing behavior

**Run this first:**
```bash
cd cactus-integration/scripts
python quick_validate.py --profile ../profiles/cactus_profile.json
```

**Expected output:**
```
✓ Loaded (dim=768)
✓ Embeddings are deterministic (diff=1.23e-07)
✓ Profile loaded
✓ Profile structure valid
✓ Dimensions correct

Testing Cluster Assignments:
  simple     → Cluster 2 (conf=0.876)
              → Best model: gemma-270m (error=45.23%)
  complex    → Cluster 0 (conf=0.912)
              → Best model: qwen-1.7b (error=18.45%)
  ...

✓ Basic validation passed
```

**If this passes:** Your profile is structurally correct and uses valid embeddings.

**If this fails:** Fix profile generation before continuing.

---

### Level 2: Full Validation with Cactus (Requires GGUF Model) 🔥

**What it checks:**
- HuggingFace vs Cactus embedding similarity
- Cluster assignment consistency
- Per-query routing decisions

**Prerequisites:**
1. Download Nomic GGUF model:
```bash
# Option A: From HuggingFace
wget https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf
mv nomic-embed-text-v1.5.f16.gguf models/

# Option B: Quantized (smaller, faster)
wget https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf
mv nomic-embed-text-v1.5.Q8_0.gguf models/
```

2. Build Cactus library (if testing locally):
```bash
# See cactus-integration/README.md for full instructions
cd ~/cactus
./aws/build_shared_lib.sh
export LD_LIBRARY_PATH=~/cactus/build/cactus:$LD_LIBRARY_PATH
```

**Run full validation:**
```bash
python validate_embedding_consistency.py \
    --mode hf-vs-cactus \
    --cactus-model models/nomic-embed-text-v1.5.f16.gguf \
    --profile ../profiles/cactus_profile.json
```

**Expected output:**
```
Generating HuggingFace embeddings...
✓ Generated 15 HF embeddings

Generating Cactus embeddings...
✓ Generated 15 Cactus embeddings

Per-Query Similarity Analysis:
0.998234 - EXCELLENT  - What is 2+2?...
0.997891 - EXCELLENT  - Explain quantum physics in detail...
0.996543 - EXCELLENT  - Write Python code to sort a list...
...

OVERALL RESULTS
Average Similarity: 0.997456
Minimum Similarity: 0.995123
Maximum Similarity: 0.998901

🎉 EXCELLENT! Average similarity = 0.9975 >= 0.99
✓ Your profile will work perfectly on phones!

VALIDATING ROUTER PROFILE CLUSTER CENTERS
✓ Query 1: Cluster 2 (both) - What is 2+2?...
✓ Query 2: Cluster 0 (both) - Explain quantum physics in detail...
✓ Query 3: Cluster 1 (both) - Write Python code to sort a list...
✓ Query 4: Cluster 3 (both) - Translate this to Spanish: Hello world...
✓ Query 5: Cluster 0 (both) - What are the symptoms of diabetes?...

🎉 PERFECT! All 5 queries assigned to same clusters
✓ Your router will behave identically on PC and phone!
```

---

### Level 3: On-Device Testing (Gold Standard) 🏆

**What it checks:**
- Actual runtime behavior on target device
- Real Cactus model loading
- End-to-end routing decisions

**Test on Android:**
```kotlin
// Load router profile
val router = AuroraRouter.load(
    profilePath = "cactus_profile.json",
    models = listOf(
        ModelInfo("gemma-270m", "weights/gemma-270m.gguf", 172),
        ModelInfo("qwen-1.7b", "weights/qwen-1.7b.gguf", 1161),
        // ... other models
    )
)

// Test routing
val testQueries = listOf(
    "What is 2+2?",
    "Explain quantum physics",
    "Write Python quicksort"
)

for (query in testQueries) {
    val result = router.route(query, costPreference = 0.5)
    println("Query: $query")
    println("  → Model: ${result.modelId}")
    println("  → Cluster: ${result.clusterId}")
    println("  → Confidence: ${result.confidence}")
}
```

**Validate against training expectations:**
- Simple queries → small models (gemma-270m, smollm-360m)
- Complex queries → large models (qwen-1.7b, smollm-1.7b)
- Coding queries → tool-capable models (lfm2-1.2b-tools)

---

## Interpreting Results

### Similarity Score Thresholds

| Similarity | Status | Meaning | Action |
|-----------|--------|---------|--------|
| ≥ 0.99 | **EXCELLENT** 🎉 | Embeddings nearly identical | ✓ Deploy immediately |
| 0.95-0.99 | **GOOD** ✓ | Minor variations acceptable | ✓ Safe to deploy |
| 0.90-0.95 | **ACCEPTABLE** ⚠️ | Some routing inaccuracies expected | Consider retraining |
| < 0.90 | **POOR** ✗ | Significant mismatches | ✗ Must retrain |

### Common Issues & Solutions

#### Issue 1: Low Similarity (< 0.95)

**Cause:** Quantization differences between FP32 (training) and FP16/INT8 (runtime)

**Solution:**
```bash
# Retrain using Cactus embeddings on ARM
# See cactus-integration/README.md for AWS Graviton setup
```

#### Issue 2: Cluster Assignment Mismatches

**Cause:** Embedding model version mismatch or different preprocessing

**Solution:**
1. Verify same model version: `nomic-ai/nomic-embed-text-v1.5`
2. Check preprocessing matches (see below)
3. Retrain with exact runtime embedding setup

#### Issue 3: 79.7% Noise Ratio (HDBSCAN)

**Cause:** HDBSCAN marked most samples as outliers

**Problem:** Queries won't have strong cluster assignments → unreliable routing

**Solution:**
```bash
# Use KMeans instead (guarantees all queries get assigned)
python scripts/train_kmeans.py \
    --n-samples 2000 \
    --n-clusters 7 \
    --output ../profiles/cactus_kmeans_profile.json
```

---

## Ensuring Exact Embedding Match

### Training Code (train_cactus_profile.py)
```python
# Text preprocessing
texts = ["search_document: " + text for text in raw_texts]

# Tokenization
inputs = tokenizer(texts, padding=True, truncation=True,
                  max_length=256, return_tensors="pt")

# Model inference
outputs = model(**inputs)

# Mean pooling
mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
summed = torch.sum(outputs.last_hidden_state * mask, dim=1)
counts = torch.clamp(mask.sum(dim=1), min=1e-9)
embeddings = summed / counts

# L2 normalization
embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
```

### Runtime Code (Must Match!)
```kotlin
// Kotlin/Cactus
val embedder = CactusEmbedder("nomic-v1.5.gguf")

// CRITICAL: Same preprocessing!
val text = "search_document: " + userQuery
val embedding = embedder.embed(text, normalize = true)
```

**Key Requirements:**
1. ✓ Same prefix: `"search_document: "`
2. ✓ Same max length: 256 tokens
3. ✓ Same pooling: Mean pooling with attention mask
4. ✓ Same normalization: L2 norm
5. ✓ Same model: `nomic-embed-text-v1.5`

---

## Quick Decision Tree

```
Start: Do you have a router profile?
│
├─ No → Run train_cactus_profile.py first
│
├─ Yes → Run quick_validate.py
    │
    ├─ Failed → Fix profile generation, retry
    │
    ├─ Passed → Do you have Cactus GGUF model?
        │
        ├─ No → Download from HuggingFace
        │      → Run validate_embedding_consistency.py
        │
        ├─ Yes → Run validate_embedding_consistency.py
            │
            ├─ Similarity < 0.95 → Retrain with Cactus on ARM
            │
            ├─ Similarity ≥ 0.95 → Deploy to phone
                │
                └─ Test on device → Monitor routing decisions
```

---

## FAQ

### Q: Can I skip Cactus validation and just test on phone?

**A:** Yes, but risky. If embeddings don't match, you'll waste time debugging on device. Better to validate first.

### Q: What if I don't have access to ARM/Graviton?

**A:** Test locally with x86 Cactus build. Similarity should still be >0.95. If not, consider cloud ARM (AWS t4g.micro = $6/month).

### Q: Should I use HDBSCAN or KMeans?

**A:** For routing, **KMeans is better** because:
- No noise → every query gets assigned
- More stable cluster assignments
- Faster inference (<5ms vs 10-15ms)

Your HDBSCAN profile has 79.7% noise which is problematic.

### Q: What about other embedding models?

**A:** Nomic v1.5 is recommended because:
- Small (138M params)
- Fast on mobile ARM
- Good quality (768-dim)
- Available in GGUF format

Alternatives:
- `all-MiniLM-L6-v2` (384-dim, faster, lower quality)
- `bge-small-en-v1.5` (384-dim, good balance)
- `gte-small` (384-dim, multilingual)

### Q: How do I know if my router is working correctly on the phone?

**A:** Log routing decisions and validate:

```kotlin
val router = AuroraRouter.load(profile, models)

// Test with known queries
val simpleQuery = "What is 2+2?"
val complexQuery = "Explain quantum entanglement in detail"

val simpleResult = router.route(simpleQuery, costPreference = 0.5)
val complexResult = router.route(complexQuery, costPreference = 0.5)

// Validate expectations
assert(simpleResult.modelId in listOf("gemma-270m", "smollm-360m")) {
    "Simple query should route to small model, got ${simpleResult.modelId}"
}

assert(complexResult.modelId in listOf("qwen-1.7b", "smollm-1.7b")) {
    "Complex query should route to large model, got ${complexResult.modelId}"
}
```

---

## Summary Checklist

Before deploying your router to production:

- [ ] **Level 1:** Run `quick_validate.py` → Passes
- [ ] **Level 2:** Run `validate_embedding_consistency.py` → Similarity ≥ 0.95
- [ ] **Level 3:** Test on actual device → Routing matches expectations
- [ ] **Noise Check:** If using HDBSCAN, noise ratio < 50%
- [ ] **Model Coverage:** All 12 models have error rates in profile
- [ ] **Capability Filtering:** Router respects model capabilities (text/vision/tools)
- [ ] **Cost Preference:** Test with different λ values (0.0, 0.5, 1.0)

---

## Need Help?

**Low similarity scores:** Retrain with Cactus embeddings on ARM
**Cluster mismatches:** Check preprocessing and normalization
**High noise ratio:** Switch to KMeans clustering
**Other issues:** Check GitHub issues or documentation

**Golden rule:** When in doubt, validate on actual device!
