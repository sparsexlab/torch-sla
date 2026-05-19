# Contributing to torch-sla

Thank you for your interest in contributing to torch-sla!

## 📁 Project Structure

```
torch-sla/
├── torch_sla/              # Source code
│   ├── backends/           # Backend implementations (scipy, torch, etc.)
│   └── distributed/        # Distributed computing support
├── benchmarks/             # Performance benchmarking scripts
│   └── benchmark_*.py
├── results/                # Benchmark results (auto-generated)
│   └── benchmark_*/
├── examples/               # Usage examples
├── tests/                  # Unit tests
└── docs/                   # Sphinx documentation
```

## 🎯 Key Conventions

### File Organization

| Type | Location | Example |
|------|----------|---------|
| Benchmark script | `benchmarks/benchmark_<feature>.py` | `benchmark_determinant.py` |
| Benchmark results | `results/benchmark_<feature>/` | `results/benchmark_determinant/` |
| Example script | `examples/<feature>.py` | `determinant.py` |
| Test file | `tests/test_<module>.py` | `test_sparse_tensor.py` |

### Naming Conventions

- **Classes**: PascalCase (`SparseTensor`, `DSparseMatrix`)
- **Functions**: snake_case (`sparse_solve()`, `det()`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_TOL`, `MAX_ITER`)
- **Private**: Leading underscore (`_validate()`, `_internal()`)

## 🔧 Development Workflow

### 1. Adding a New Feature

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Implement in torch_sla/
vim torch_sla/sparse_tensor.py

# 3. Add tests
vim tests/test_sparse_tensor.py
pytest tests/test_sparse_tensor.py

# 4. Add example
vim examples/my_feature.py

# 5. Add benchmark
vim benchmarks/benchmark_my_feature.py
python benchmarks/benchmark_my_feature.py

# 6. Update docs
vim docs/source/examples.rst

# 7. Update TODO.md
vim TODO.md

# 8. Commit and push
git add .
git commit -m "feat(sparse_tensor): add my feature"
git push origin feature/my-feature
```

### 2. Benchmark Script Template

```python
#!/usr/bin/env python3
"""
Benchmark for <feature> performance.
Results are saved to: results/benchmark_<feature>/
"""
import torch
import time
from pathlib import Path
import matplotlib.pyplot as plt
from torch_sla import SparseTensor

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "benchmark_<feature>"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def benchmark_feature():
    """Benchmark the feature."""
    # Your benchmark code
    pass

def plot_results(data):
    """Plot and save results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plotting code
    plt.savefig(OUTPUT_DIR / "results.png", dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    results = benchmark_feature()
    plot_results(results)
    print(f"\nResults saved to {OUTPUT_DIR}")
```

### 3. Documentation Standards

Use Google-style docstrings:

```python
def det(self) -> torch.Tensor:
    """
    Compute the determinant of the sparse matrix.
    
    This method uses sparse LU decomposition on CPU and dense
    conversion on CUDA.
    
    Returns:
        Determinant value as a scalar tensor.
        
    Example:
        >>> dense = torch.tensor([[4., -1., 0.], [-1., 4., -1.], [0., -1., 4.]])
        >>> A = SparseTensor.from_dense(dense)
        >>> det = A.det()
        
    Note:
        CPU is 3-180x faster than CUDA for sparse matrices.
        Use `.cpu().det()` even for CUDA tensors.
        
    Warning:
        Determinant values can overflow for large matrices (n > 1000).
    """
```

## 🧪 Testing

### Running Tests

```bash
# All tests
pytest tests/

# Specific test
pytest tests/test_sparse_tensor.py::TestDeterminant

# With coverage
pytest --cov=torch_sla tests/

# Verbose
pytest -v tests/
```

### Test Structure

```python
import pytest
import torch
from torch_sla import SparseTensor

class TestDeterminant:
    """Tests for determinant computation."""
    
    def test_det_basic(self):
        """Test basic determinant computation."""
        dense = torch.tensor([[1.0, 2.0],
                              [3.0, 4.0]])
        A = SparseTensor.from_dense(dense)
        
        det = A.det()
        
        assert torch.isclose(det, torch.tensor(-2.0))
    
    @pytest.mark.skipif(not torch.cuda.is_available(), 
                       reason="CUDA not available")
    def test_det_cuda(self):
        """Test determinant on CUDA."""
        # Test implementation
        pass
```

## 📊 Benchmarking

### Running Benchmarks

```bash
# Run specific benchmark
python benchmarks/benchmark_determinant.py

# Results are saved to
ls results/benchmark_determinant/
```

### Benchmark Output

Each benchmark should produce:
- **PNG figure**: `results/benchmark_<feature>/<name>.png`
- **Console summary**: Key findings and recommendations

See `benchmarks/README.md` for detailed benchmark results.

## 📝 Commit Message Format

Use conventional commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `perf`: Performance improvement
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Maintenance

**Examples**:
```bash
feat(sparse_tensor): add determinant computation with gradient support
fix(solve): handle singular matrices correctly
docs(examples): add distributed computing example
perf(det): optimize CPU sparse LU decomposition
```

## 📦 Release Process

Releases to [PyPI](https://pypi.org/project/torch-sla/) are fully automated. Pushing a tag matching `v*` triggers `.github/workflows/publish.yml`, which builds the sdist + wheel and uploads them via PyPI Trusted Publishing (OIDC — no API token needed).

### Standard release (push tag → auto-publish)

```bash
# 1. Bump version in pyproject.toml (and anywhere else it lives)
vim pyproject.toml

# 2. Commit and tag
git add pyproject.toml
git commit -m "chore: bump version to X.Y.Z"
git tag vX.Y.Z

# 3. Push commit + tag — this triggers the publish workflow
git push origin main
git push origin vX.Y.Z
```

Watch the run at https://github.com/sparsexlab/torch-sla/actions/workflows/publish.yml. When it goes green, the new version is live on PyPI.

### Optional: also create a GitHub Release (with changelog)

The `release: published` event is kept as a secondary trigger. If you want a GitHub Release page with release notes:

1. Push the tag as above (publish workflow runs once via `push: tags`).
2. Go to https://github.com/sparsexlab/torch-sla/releases/new , pick the existing tag, write notes, click **Publish release**.

Creating a Release from an existing tag does **not** re-trigger the workflow, so PyPI won't get a duplicate upload.

### Test PyPI dry-run

Before a real release, you can publish to [Test PyPI](https://test.pypi.org/project/torch-sla/) without tagging:

1. Actions → "Publish to PyPI" → **Run workflow**
2. Set `test_pypi` to `true` and run on `main`

### Important rules

- **Never reuse a version number.** PyPI permanently reserves every version that has ever been uploaded — even deleted/yanked ones. If `X.Y.Z` was ever pushed, you must bump to a new number.
- **Tag must start with `v`** (e.g. `v0.2.2`, not `0.2.2`). The workflow trigger is `tags: ['v*']`.
- **Version in `pyproject.toml` must match the tag.** The build uses whatever `pyproject.toml` says; mismatches will publish under the wrong version and confuse users.
- **Trusted Publishing is configured on PyPI for this repo + workflow + `pypi` environment.** Renaming the workflow file, the environment, or moving the repo will break uploads until you re-add the trusted publisher on PyPI.

## 🚀 Performance Guidelines

### CPU vs CUDA

| Operation | CPU | CUDA | Recommendation |
|-----------|-----|------|----------------|
| Sparse determinant | ✅ Fast | ❌ Slow | Always use CPU |
| Sparse solve (small) | ✅ Fast | ❌ Slow | Use CPU |
| Sparse solve (large) | ⚠️ OK | ✅ Fast | Use CUDA |
| Dense operations | ⚠️ OK | ✅ Fast | Use CUDA |

### Key Findings

1. **Sparse determinant**: CPU is 3-180x faster than CUDA
   ```python
   # ❌ Slow
   det = A_cuda.det()
   
   # ✅ Fast
   det = A_cuda.cpu().det()
   ```

2. **Gradient computation**: ~100x slower than forward pass (expected)

3. **Numerical stability**: Use `float64` for better precision

## 🐛 Common Issues

### Issue: CUDA determinant is slow
**Solution**: Use CPU for sparse determinant
```python
det = A_cuda.cpu().det()  # Much faster!
```

### Issue: Import error for cuDSS
**Solution**: Install nvmath-python
```bash
pip install nvmath-python[cu12]
```

## 📚 Resources

- **Documentation**: https://www.torchsla.com/
- **GitHub**: https://github.com/walkerchi/torch-sla
- **Issues**: https://github.com/walkerchi/torch-sla/issues

## 📋 Checklist for New Features

Before submitting a PR:

- [ ] Implementation in `torch_sla/`
- [ ] Tests in `tests/` (with good coverage)
- [ ] Example in `examples/`
- [ ] Benchmark in `benchmarks/`
- [ ] Documentation in `docs/source/`
- [ ] Update `README.md` if needed
- [ ] Update `TODO.md`
- [ ] All tests pass
- [ ] Code follows style guide
- [ ] Docstrings are complete

## 💡 Tips

1. **Start small**: Begin with simple contributions
2. **Ask questions**: Open an issue if unsure
3. **Follow conventions**: Consistency is key
4. **Test thoroughly**: Include edge cases
5. **Document well**: Help others understand your code

---

Thank you for contributing to torch-sla! 🎉

