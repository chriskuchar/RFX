# RFX: Random Forests X (CPU-Only Edition)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/rfx-ml-cpu.svg)](https://pypi.org/project/rfx-ml-cpu/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2511.19493-b31b1b.svg)](https://arxiv.org/abs/2511.19493)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/)

**RFX** (Random Forests X) is a high-performance Python implementation of Breiman and Cutler's original Random Forest methodology with an optimized C++ back-end.

**This is the CPU-only version** (`rfx-ml-cpu`), built without CUDA dependencies, making it ideal for systems without GPUs or lightweight installations.

> **Note:** For GPU acceleration and QLORA compression with large datasets, use the [`rfx-ml`](https://pypi.org/project/rfx-ml/) package instead.

## Key Features

- **Complete classification**: Out-of-bag error, confusion matrices, class probabilities
- **Local importance**: Per-sample feature importance (similar to SHAP, built-in)
- **Proximity matrices**: Pairwise sample similarities for outlier detection and visualization
- **CPU-optimized**: Fast multi-threaded C++ implementation
- **No CUDA required**: Works on any system without GPU dependencies
- **Interactive visualization**: Python-native rfviz with 3D MDS and parallel coordinates

## Installation

**CPU-Only Version** (this package):

```bash
pip install rfx-ml-cpu
```

**GPU-Enabled Version** (if you have CUDA):

```bash
pip install rfx-ml
```

**Note:** These packages are mutually exclusive. Both provide the `rfx` module. Choose based on your hardware:
- CPU-only system or want minimal dependencies? → `rfx-ml-cpu`
- Have a GPU and want acceleration? → `rfx-ml`

**Prerequisites:** CMake 3.12+, Python 3.8+, C++ compiler with C++17 support. No CUDA required!

The `pip install` command will automatically build from source. Make sure you have the prerequisites installed before running pip.

## Quick Start

```python
import numpy as np
import rfx as rf

# Load sample data
X, y = rf.load_wine()

# Train Random Forest (CPU-only)
model = rf.RandomForestClassifier(
    ntree=100,
    use_gpu=False,  # Required: CPU-only package
    compute_importance=True,
    compute_local_importance=True,
    compute_proximity=True
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

## CPU Performance

RFX-CPU uses highly optimized multi-threaded C++ code:

```python
# Automatic multi-threading (uses all CPU cores by default)
model = rf.RandomForestClassifier(
    ntree=500,
    use_gpu=False,
    n_threads_cpu=0,  # Auto-detect CPU cores
    compute_proximity=True
)

model.fit(X, y)
```

**When to use this package:**
- No GPU available
- Lightweight installation needed
- Small to medium datasets (<50K samples)
- CPU-only deployment environments

**For large datasets (>50K samples) with proximity matrices**, consider [`rfx-ml`](https://pypi.org/project/rfx-ml/) with GPU acceleration for significantly faster performance.

## Switching Between Versions

Both packages use the same `import rfx` statement. To switch:

```bash
# Switch to GPU version
pip uninstall rfx-ml-cpu
pip install rfx-ml
```

## Documentation

For complete documentation, examples, and advanced usage, visit:
- **GitHub**: https://github.com/chriskuchar/RFX
- **Full README**: https://github.com/chriskuchar/RFX/blob/main/README.md
- **GPU Version**: https://pypi.org/project/rfx-ml/

## License

MIT License - see [LICENSE](https://github.com/chriskuchar/RFX/blob/main/LICENSE) file for details.

## Links

- **Source Code**: https://github.com/chriskuchar/RFX
- **Bug Reports**: https://github.com/chriskuchar/RFX/issues
- **Documentation**: https://github.com/chriskuchar/RFX/blob/main/README.md

