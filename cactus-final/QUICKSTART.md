# 🚀 Quick Start Guide

## On Your Current Machine (x86 Windows)

You can test the pipeline structure using mock embeddings:

```bash
# 1. Install dependencies (mock mode)
pip install datasets numpy pandas scikit-learn hdbscan tqdm sentence-transformers

# 2. Test bindings (will show "not available" - that's expected)
python bindings/test_bindings.py

# 3. View model configuration
python training/config.py

# 4. Generate test profile with mock embeddings
python training/generate_profile.py \
    --mock-embeddings \
    --output profiles/test_profile.json
```

**⚠️ Important:** Mock profiles use HuggingFace embeddings, NOT Cactus. Only use for testing the pipeline!

---

## On Your Mac (ARM) - For Production Profiles

### One-Time Setup

```bash
# 1. Run setup script
cd auroraai-router/cactus-final
chmod +x setup.sh
./setup.sh

# 2. Download embedding model
cd ../../cactus
./cli/cactus download LiquidAI/LFM2-350M

# This downloads to: cactus/weights/LFM2-350M/
```

### Generate Production Profile

```bash
cd auroraai-router/cactus-final

# Generate profile with real Cactus embeddings
python training/generate_profile.py \
    --use-cactus \
    --model-path ../../cactus/weights/LFM2-350M/model.gguf \
    --output profiles/production_profile.json
```

**⏱️ Time:** 10-30 minutes depending on your Mac

**📁 Output:** `profiles/production_profile.json` (~100KB)

---

## Verify Profile

After generation, check the profile:

```bash
# View profile metadata
python -c "import json; p=json.load(open('profiles/production_profile.json')); print(json.dumps(p['metadata'], indent=2))"

# Check profile size
ls -lh profiles/production_profile.json
```

You should see:
- ✅ `is_mock: false` (if using Cactus)
- ✅ `embedding_model: "lfm2-350m"` (or your chosen model)
- ✅ File size ~100-200KB

---

## Next Steps

1. **Integrate with router:** Use the profile with AuroraAI Router SDK
2. **Replace error simulation:** Add real Cactus inference for error rate calculation
3. **Test routing:** Create test queries and validate routing decisions

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Cactus library not found" | Run `./setup.sh` or build Cactus manually: `cd cactus && ./apple/build.sh` |
| "ARM architecture required" | Use `--mock-embeddings` on x86, or run on Mac |
| "Model file not found" | Download with: `cd cactus && ./cli/cactus download LiquidAI/LFM2-350M` |
| Out of memory | Reduce `samples_per_topic` in `training/config.py` |

---

## File Structure

```
cactus-final/
├── bindings/cactus_bindings.py    # C library wrapper
├── training/generate_profile.py   # Main script
├── training/config.py             # 12 models config
├── profiles/                      # Output profiles here
└── README.md                      # Full documentation
```

That's it! 🎉
