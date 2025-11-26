# RFX: Random Forests X

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/rfx-ml.svg)](https://pypi.org/project/rfx-ml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2511.19493-b31b1b.svg)](https://arxiv.org/abs/2511.19493)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**RFX** (Random Forests X) is a high-performance Python implementation of Breiman and Cutler's original Random Forest methodology with GPU acceleration and QLORA compression.

## Key Features

- **Complete classification**: Out-of-bag error, confusion matrices, class probabilities
- **Local importance**: Per-sample feature importance (similar to SHAP, built-in)
- **Proximity matrices**: Pairwise sample similarities for outlier detection and visualization
- **QLORA compression**: 12,500× memory reduction (80GB → 6.4MB) for large-scale proximity analysis
- **Full GPU acceleration**: CUDA for trees, importance, and proximity matrices
- **Interactive visualization**: Python-native rfviz with 3D MDS and parallel coordinates

**Result:** Proximity-based workflows now scale to 200K–1M+ samples.

## Installation

**GPU-Enabled Version** (supports both GPU and CPU fallback):

```bash
pip install rfx-ml
```

**CPU-Only Version** (lightweight, no CUDA dependencies):

```bash
pip install rfx-ml-cpu
```

**Note:** These packages are mutually exclusive. Both provide the `rfx` module. Choose based on your hardware:
- Have a GPU and want acceleration? → `rfx-ml`
- CPU-only system or want minimal dependencies? → `rfx-ml-cpu`

**Prerequisites:** CMake 3.12+, Python 3.7+, CUDA toolkit 11.0+ (required for building; GPU usage optional at runtime), C++ compiler with C++17 support.

The `pip install` command will automatically build from source. Make sure you have the prerequisites installed before running pip.

## Quick Start

```python
import numpy as np
import rfx as rf

# Load sample data
X, y = rf.load_wine()

# Train Random Forest
model = rf.RandomForestClassifier(
    ntree=100,
    compute_importance=True,
    compute_local_importance=True,
    compute_proximity=True,
    use_gpu=False  # Set to True for GPU acceleration
)

model.fit(X, y)

# Get predictions and metrics
oob_error = model.get_oob_error()
print(f"OOB Error: {oob_error:.4f}")

predictions = model.predict(X)
importance = model.feature_importances_()
local_imp = model.get_local_importance()

# Interactive visualization
rf.rfviz(
    rf_model=model,
    X=X,
    y=y,
    output_file="rfviz_example.html"
)
```

## GPU Acceleration & QLORA

For large datasets, enable GPU acceleration and QLORA compression:

```python
# Large-scale proximity analysis with QLORA
model = rf.RandomForestClassifier(
    ntree=500,
    use_gpu=True,
    compute_proximity=True,
    use_qlora=True,
    rank=32,  # Low-rank approximation
    quant_mode="int8"
)

model.fit(X, y)

# Get low-rank factors (memory efficient)
A, B, rank = model.get_lowrank_factors()

# Compute MDS directly from factors (no reconstruction!)
mds_coords = model.compute_mds_from_factors(k=3)
```

**Memory savings:** 100K samples: 74.5 GB (full matrix) → 19 MB (QLORA rank-100) = 4000× compression.

## Documentation

For complete documentation, examples, and advanced usage, visit:
- **GitHub**: https://github.com/chriskuchar/RFX
- **Full README**: https://github.com/chriskuchar/RFX/blob/main/README.md

## License

MIT License - see [LICENSE](https://github.com/chriskuchar/RFX/blob/main/LICENSE) file for details.

## Links

- **Source Code**: https://github.com/chriskuchar/RFX
- **Bug Reports**: https://github.com/chriskuchar/RFX/issues
- **Documentation**: https://github.com/chriskuchar/RFX/blob/main/README.md

