# UV Migration Guide

The project now uses **uv** for dependency management instead of raw pip + requirements.txt.

## What Changed

### Files Added
- `pyproject.toml` - Project metadata and dependencies
- `.python-version` - Python version specification (3.10)

### Files Updated
- `README.md` - Updated all commands to use `uv run`
- `QUICKSTART.md` - Updated installation instructions
- `setup.sh` - Added uv detection with pip fallback

### Files Kept
- `requirements.txt` - Still available for pip-based workflows

---

## Benefits of UV

✅ **Faster installs** - 10-100x faster than pip
✅ **Better dependency resolution** - Consistent across environments
✅ **Lock files** - `uv.lock` ensures reproducible installs
✅ **Virtual env management** - Automatic venv creation
✅ **Drop-in replacement** - Works with existing pip workflows

---

## Usage

### Install UV (One-Time)

**Mac/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Via pip:**
```bash
pip install uv
```

---

### Install Dependencies

**Core dependencies only:**
```bash
uv sync
```

**With optional extras:**
```bash
# Mock embeddings (for x86 testing)
uv sync --extra mock

# Visualization tools
uv sync --extra viz

# Development tools
uv sync --extra dev

# Everything
uv sync --all-extras
```

---

### Run Scripts

**Using uv run:**
```bash
uv run python bindings/test_bindings.py
uv run python training/config.py
uv run python training/generate_profile.py --mock-embeddings
```

**Direct execution (if venv activated):**
```bash
# Activate venv first
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

# Then run normally
python bindings/test_bindings.py
```

---

## Migrating Existing Setup

If you already have the project set up with pip:

```bash
cd auroraai-router/cactus-final

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (creates new venv in .venv/)
uv sync

# Optional: Remove old venv if you had one
rm -rf venv/ env/
```

---

## Dependency Groups

### Core (default)
Required for profile generation:
- datasets
- numpy
- pandas
- scikit-learn
- hdbscan
- tqdm

### [mock]
For x86 testing without Cactus:
- sentence-transformers

### [viz]
For visualization (future):
- matplotlib
- seaborn
- plotly
- umap-learn

### [dev]
For development:
- pytest
- black
- ruff

---

## Fallback to Pip

The project still supports pip:

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install as editable package
pip install -e .

# With optional dependencies
pip install -e ".[mock]"
pip install -e ".[viz,dev]"
```

The `setup.sh` script automatically detects uv and falls back to pip if not available.

---

## Common Commands

| Task | UV Command | Pip Equivalent |
|------|------------|----------------|
| Install deps | `uv sync` | `pip install -r requirements.txt` |
| Add dependency | `uv add <package>` | Edit requirements.txt + `pip install` |
| Remove dependency | `uv remove <package>` | Edit requirements.txt + `pip uninstall` |
| Run script | `uv run python script.py` | `python script.py` |
| Update deps | `uv sync --upgrade` | `pip install --upgrade -r requirements.txt` |
| Lock deps | `uv lock` | N/A (manual) |

---

## Lock File

UV generates `uv.lock` which pins exact versions:

```bash
# Commit uv.lock for reproducibility
git add uv.lock
git commit -m "Add uv lock file"

# On other machines, uv sync will use locked versions
uv sync
```

---

## Troubleshooting

### UV not found after install
```bash
# Reload shell
source ~/.bashrc  # or ~/.zshrc

# Or add to PATH manually
export PATH="$HOME/.cargo/bin:$PATH"
```

### Virtual environment location
UV creates venvs in `.venv/` by default:
```bash
ls -la .venv/

# Activate manually if needed
source .venv/bin/activate
```

### Prefer system Python
```bash
# Use specific Python version
uv sync --python 3.10

# Or with pyenv
uv sync --python $(pyenv which python)
```

---

## Migration Checklist

- [x] Create `pyproject.toml`
- [x] Create `.python-version`
- [x] Update README.md
- [x] Update QUICKSTART.md
- [x] Update setup.sh
- [x] Keep requirements.txt for compatibility
- [x] Document migration in UV_MIGRATION.md

---

## Resources

- **UV Docs:** https://docs.astral.sh/uv/
- **UV GitHub:** https://github.com/astral-sh/uv
- **Announcement:** https://astral.sh/blog/uv

---

## Summary

The project now uses **uv** for faster, more reliable dependency management while maintaining backward compatibility with pip. All documentation has been updated to use `uv run` commands.

**Install uv and run:**
```bash
uv sync
uv run python training/config.py
```

**Or stick with pip:**
```bash
pip install -r requirements.txt
python training/config.py
```

Both work! 🎉
