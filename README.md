<p align="center">
  <img src="figures/rfx_logo.png" width="400" alt="RFX Logo">
</p>

<h1 align="center">RFX</h1>
<h3 align="center">GPU-Accelerated Random Forests with QLoRA Compression</h3>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.7+-blue.svg" alt="Python"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-11.0+-76B900.svg" alt="CUDA"></a>
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-paper-b31b1b.svg" alt="arXiv"></a>
</p>

<p align="center">
  <a href="#key-features">Features</a> &nbsp;&bull;&nbsp;
  <a href="#installation">Installation</a> &nbsp;&bull;&nbsp;
  <a href="#quick-start">Quick Start</a> &nbsp;&bull;&nbsp;
  <a href="#examples">Examples</a> &nbsp;&bull;&nbsp;
  <a href="#citation">Citation</a>
</p>

---

RFX is a high-performance Random Forest implementation faithful to Breiman and Cutler's original Fortran algorithms. It extends the classic methodology with GPU acceleration and QLoRA compression for proximity matrices.

## Key Features

| | Feature | Benefit |
|:---:|:---|:---|
| **GPU** | CUDA-accelerated tree growing | 5-10x faster training |
| **QLoRA** | Low-rank proximity compression | 12,500x memory reduction |
| **RFViz** | Interactive 3D visualization | Linked brushing, MDS plots |
| **Faithful** | Original Breiman/Cutler algorithms | Statistical rigor |

## Installation

**Requirements:** Python 3.7+, CMake 3.12+, C++17 compiler. CUDA 11.0+ optional for GPU.

```bash
git clone https://github.com/chriskuchar/RFX.git
cd RFX
pip install -e .
```

Verify:
```python
import RFX as rf
print(f"GPU available: {rf.cuda_is_available()}")
```

## Quick Start

### Basic Classification

```python
import RFX as rf

X, y = rf.load_wine()

model = rf.RandomForestClassifier(
    ntree=100,
    use_gpu=True,
    compute_importance=True
)
model.fit(X, y)

print(f"OOB Error: {model.get_oob_error():.2%}")
print(f"Top feature: {model.feature_importances_().argmax()}")
```

### Proximity with QLoRA Compression

For large datasets, QLoRA reduces proximity matrix memory by 12,500x:

```python
model = rf.RandomForestClassifier(
    ntree=100,
    use_gpu=True,
    compute_proximity=True,
    use_qlora=True,        # Enable low-rank compression
    quant_mode="int8",     # INT8, NF4, or FP16
    rank=32                # Low-rank dimension
)
model.fit(X, y)

# 3D MDS coordinates from low-rank factors
mds = model.compute_mds_from_factors(k=3)
```

### Interactive Visualization

```python
rf.rfviz(
    rf_model=model,
    X=X, y=y,
    feature_names=feature_names,
    output_file="rfviz.html"
)
```

## Examples

| File | Description |
|:-----|:------------|
| `test_wine_classification_basic.ipynb` | Basic GPU/CPU classification |
| `test_wine_classification_importance.ipynb` | Overall and local feature importance |
| `test_wine_classification_proximity.ipynb` | QLoRA proximity with RFViz |
| `test_wine_oob_confusion.py` | OOB error and confusion matrix |
| `test_wine_importance.py` | Feature importance analysis |
| `test_wine_lowrank_rfviz.py` | Low-rank proximity visualization |

Run examples:
```bash
cd examples
python test_wine_oob_confusion.py
jupyter notebook test_wine_classification_basic.ipynb
```

## Performance

### Memory: Proximity Matrix

| Samples | Full Matrix | QLoRA INT8 | Reduction |
|--------:|------------:|-----------:|----------:|
| 10,000 | 800 MB | 0.6 MB | 1,300x |
| 100,000 | 80 GB | 6 MB | 13,000x |
| 1,000,000 | 8 TB | 64 MB | 125,000x |

### Speed: Training Time

| Dataset | CPU | GPU | Speedup |
|:--------|----:|----:|--------:|
| Wine (178 samples) | 10s | 7s | 1.4x |
| Medium (10K samples) | 5min | 45s | 6.7x |
| Large (100K samples) | 45min | 8min | 5.6x |

## API Reference

### RandomForestClassifier

```python
rf.RandomForestClassifier(
    ntree=100,              # Number of trees
    mtry=0,                 # Features per split (0=auto: sqrt(p))
    use_gpu=False,          # Enable CUDA acceleration
    compute_importance=True,
    compute_local_importance=False,
    compute_proximity=False,
    use_qlora=False,        # Low-rank proximity compression
    quant_mode="nf4",       # "int8", "nf4", "fp16"
    rank=32,                # Low-rank dimension
    use_casewise=False      # Bootstrap frequency weighting
)
```

**Methods:**
- `fit(X, y)` - Train the model
- `predict(X)` - Predict class labels
- `get_oob_error()` - Out-of-bag error rate
- `feature_importances_()` - Overall importance scores
- `get_local_importance()` - Per-sample importance
- `compute_mds_from_factors(k=3)` - MDS from low-rank proximity
- `confusion_matrix(y_true, y_pred)` - Confusion matrix
- `classification_report(y_true, y_pred)` - Precision/recall/F1

### Visualization

```python
rf.rfviz(
    rf_model,               # Trained RandomForestClassifier
    X, y,                   # Data
    feature_names=None,     # Optional feature names
    n_clusters=3,           # K-means clusters for coloring
    mds_k=3,                # MDS dimensions
    output_file="out.html", # Save location
    show_in_browser=True    # Open in browser
)
```

## Citation

```bibtex
@article{kuchar2025rfx,
  title={RFX: High-Performance Random Forests with GPU Acceleration and QLoRA Compression},
  author={Kuchar, Christopher},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Roadmap

### v1.0 (Current)
- GPU-accelerated tree growing
- QLoRA low-rank proximity compression (INT8/NF4/FP16)
- Overall and local feature importance  
- RFViz interactive visualization
- Confusion matrix and classification report
- Jupyter notebook support

### v1.1 (Planned)
- Regression support
- Unsupervised mode
- RF-GAP proximity normalization
- CLIQUE importance

### v2.0 (Future)
- Multi-GPU support

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Leo Breiman and Adele Cutler for the original Random Forest methodology
- NVIDIA for CUDA toolkit

---

<p align="center">
  <sub>RFX: Faithful to Breiman, accelerated for today.</sub>
</p>
