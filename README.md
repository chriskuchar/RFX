# RFX: Random Forests X

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

**RFX** (Random Forests X, where X represents compression/quantization) is a high-performance and production-ready Python implementation of Breiman and Cutler's original Random Forest classification methodology, faithfully following all algorithms from the original Fortran code with no shortcuts on methodology whatsoever. This work aims to honor the legacy of Leo Breiman and Adele Cutler by ensuring their Random Forest methodology is not forgotten and remains accessible to modern researchers.

## 🎯 Key Features

RFX v1.0 provides complete classification capabilities with modern enhancements:

### Core Classification Features (Original Fortran)
- ✅ **Complete classification**: Out-of-bag error estimation, confusion matrices, and class probability predictions
- ✅ **Proximity matrices**: Pairwise sample similarities enabling outlier detection, clustering, and visualization
- ✅ **Overall and local importance**: Feature-level and sample-specific importance measures
- ✅ **Case-wise analysis**: Bootstrap weighting and out-of-bag evaluation from unreleased Fortran extensions
- ✅ **Interactive visualization**: Python-native rfviz with 3D MDS, parallel coordinates, and linked brushing

### Modern Enhancements
- 🚀 **GPU acceleration**: CUDA implementations for tree growing, importance computation, and proximity matrices
- 💾 **QLORA proximity compression**: Quantized low-rank adaptation reducing 80GB matrices to 6.4MB (12,500× compression) with 99% geometric structure preservation
- 📦 **CPU TriBlock proximity**: Upper-triangle + block-sparse storage achieving 2.7× memory reduction with lossless quality
- 🎨 **GPU-accelerated MDS**: Power iteration method computing 3D embeddings directly from low-rank factors
- ⚡ **SM-aware GPU batching**: Automatic batch sizing based on GPU architecture, achieving 95% GPU utilization

## 📦 Installation

### Prerequisites

- **CMake** 3.12 or higher
- **Python** 3.7+ (tested up to 3.13)
- **CUDA toolkit** 11.0+ (for GPU support, optional but recommended)
- **C++ compiler** with C++17 support (GCC 7+, Clang 5+)
- **OpenMP** (usually included with compiler)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/chriskuchar/RFX.git
cd RFX

# Install with pip (handles CMake build automatically)
pip install -e .

# Or install with visualization dependencies
pip install -e ".[viz]"

# Or install with all optional dependencies
pip install -e ".[viz,examples]"
```

The `pip install` command will automatically:
1. Configure CMake
2. Build the C++/CUDA extensions
3. Install the Python package

### Manual Build (Alternative)

If you prefer to build manually:

```bash
# Create build directory
mkdir -p build && cd build

# Configure with CMake
cmake ..

# Build (uses all available CPU cores)
make -j$(nproc)

# Install Python package
cd ..
pip install -e .
```

### Verify Installation

```python
import RFX as rf
print(f"RFX version: {rf.__version__}")
print(f"CUDA enabled: {rf.__cuda_enabled__}")
```

## 🚀 Quick Start

### Basic Classification

```python
import numpy as np
import RFX as rf

# Load your data (or use built-in dataset)
X = np.random.randn(1000, 10).astype(np.float32)
y = np.random.randint(0, 3, 1000).astype(np.int32)

# Create and train model
model = rf.RandomForestClassifier(
    ntree=100,
    compute_importance=True,
    compute_proximity=False,
    use_gpu=False  # Set to True for GPU acceleration
)

model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Get OOB error
oob_error = model.oob_error()
print(f"OOB Error: {oob_error:.4f}")

# Get confusion matrix
confusion = model.confusion_matrix()
print("Confusion Matrix:")
print(confusion)

# Get overall feature importance
importance = model.overall_importance()
print("Top 3 Features:", np.argsort(importance)[-3:][::-1])
```

### GPU Acceleration

```python
# Enable GPU with automatic batch sizing
model = rf.RandomForestClassifier(
    ntree=500,
    use_gpu=True,
    batch_size=0,  # 0 = auto-scale based on GPU (recommended)
    compute_importance=True
)

model.fit(X, y)
```

### Proximity Matrices with QLORA Compression

```python
# GPU proximity with QLORA compression (memory efficient)
model = rf.RandomForestClassifier(
    ntree=100,
    use_gpu=True,
    compute_proximity=True,
    use_qlora=True,  # Enable QLORA compression
    quant_mode="int8",  # INT8 quantization (recommended)
    rank=32  # Low-rank approximation rank
)

model.fit(X, y)

# Get low-rank factors (memory efficient)
A, B = model.get_lowrank_factors()
print(f"Low-rank factors shape: A={A.shape}, B={B.shape}")

# Reconstruct proximity matrix (if needed)
proximity = A @ B.T

# Or compute MDS directly from factors (no reconstruction needed)
mds_coords = model.compute_mds_3d_from_factors()
print(f"MDS coordinates shape: {mds_coords.shape}")
```

### CPU TriBlock Proximity (Lossless Compression)

```python
# CPU proximity with TriBlock storage (lossless, 2.7× memory reduction)
model = rf.RandomForestClassifier(
    ntree=100,
    use_gpu=False,
    compute_proximity=True,
    use_sparse=True,  # Enable block-sparse thresholding
    sparsity_threshold=0.0001  # Ultra-conservative threshold
)

model.fit(X, y)

# Get full proximity matrix (lossless quality)
proximity = model.get_proximity_matrix()
```

### Case-wise Analysis

```python
# Enable case-wise mode (bootstrap frequency weighting)
model = rf.RandomForestClassifier(
    ntree=100,
    use_casewise=True,  # Enable case-wise calculations
    compute_importance=True,
    compute_local_importance=True
)

model.fit(X, y)

# Get case-wise overall importance
importance = model.overall_importance()

# Get case-wise local importance (per-sample, per-feature)
local_imp = model.local_importance()
print(f"Local importance shape: {local_imp.shape}")  # (n_samples, n_features)
```

### Interactive Visualization (rfviz)

```python
# Generate interactive 2×2 grid visualization
rf.rfviz(
    rf_model=model,
    X=X,
    y=y,
    feature_names=["Feature 1", "Feature 2", ...],  # Optional
    output_file="rfviz_output.html",
    show_in_browser=True
)
```

This creates an interactive HTML file with:
- **Top-left**: Input features parallel coordinates
- **Top-right**: Local importance parallel coordinates
- **Bottom-left**: 3D MDS proximity plot (rotatable, zoomable)
- **Bottom-right**: Class votes heatmap (RAFT-style)

All plots have **linked brushing** - selecting samples in one plot highlights them in all others. Press **R** or **Escape** to clear selections.

## 📊 Performance

### Memory Requirements

| Dataset Size | CPU Full | CPU TriBlock | GPU INT8 (rank-32) | GPU NF4 (rank-32) |
|--------------|----------|--------------|---------------------|-------------------|
| 1,000        | 0.0 GB   | 0.0 GB       | 0.1 MB              | 0.0 MB            |
| 10,000       | 0.7 GB   | 0.3 GB       | 0.6 MB              | 0.3 MB            |
| 50,000       | 18.6 GB  | 7.5 GB       | 3.1 MB              | 1.5 MB            |
| 100,000      | 74.5 GB  | 29.8 GB      | 6.1 MB              | 3.1 MB            |
| 200,000      | 298.0 GB | 119.2 GB     | 12.2 MB             | 6.1 MB            |

**Recommendation**: Use CPU TriBlock for <50K samples, GPU INT8 for 50K+ samples.

### Speed Comparison (Wine Dataset, 500 trees)

| Task | CPU | GPU | Speedup |
|------|-----|-----|---------|
| Overall Importance | 9.96s | 7.09s | 1.4× |
| Local Importance (100 trees) | 4.16s | 22.57s | 0.18× (CPU faster) |
| Proximity (100 trees, small dataset) | 3.94s | 17.38s | 0.23× (CPU faster) |

**Note**: GPU excels at overall importance with 500+ trees. CPU is faster for local importance and proximity on small datasets. GPU advantage increases with dataset size.

## 📚 Examples

The `examples/` folder contains scripts to reproduce all results from the arXiv paper:

### Classification Examples
- `test_wine_step1_oob.py` - OOB error and confusion matrix (Table 6, Table 7)
- `test_wine_step2_importance.py` - Overall importance comparison (Table 5)
- `test_wine_step3_local_importance.py` - Local importance benchmarking

### Proximity Examples
- `test_proximity_4way.py` - 4-way proximity comparison (GPU/CPU × casewise/non-casewise)
- `test_quantization_comparison.py` - Quantization level comparison (Table 8)
- `test_final_quantization_1000trees.py` - 1000-tree validation
- `test_cpu_block_sparse.py` - CPU TriBlock proximity
- `test_block_sparse_thresholds.py` - Block-sparse threshold analysis

### Visualization Examples
- `test_oob_convergence_1k.py` - OOB convergence plot (Figure 4)
- `test_wine_gpu_rfviz.py` - RFviz 2×2 grid visualization (Figure 8)

Run any example:
```bash
cd examples
python test_wine_step1_oob.py
```

## 🔧 API Reference

### RandomForestClassifier

```python
rf.RandomForestClassifier(
    ntree=100,                    # Number of trees
    mtry=0,                       # Features per split (0 = sqrt(mdim))
    maxnode=0,                    # Max nodes per tree (0 = 2*nsample + 1)
    minndsize=1,                  # Minimum node size
    compute_proximity=False,       # Compute proximity matrix
    compute_importance=True,       # Compute overall importance
    compute_local_importance=False, # Compute local importance
    use_gpu=False,                 # Enable GPU acceleration
    use_qlora=False,               # Enable QLORA compression (GPU only)
    quant_mode="nf4",             # Quantization: "nf4", "int8", "fp16", "fp32"
    rank=100,                      # Low-rank approximation rank
    use_sparse=False,              # Enable CPU block-sparse (TriBlock)
    sparsity_threshold=1e-6,       # Block-sparse threshold
    batch_size=0,                  # GPU batch size (0 = auto-scale)
    use_casewise=False,            # Case-wise calculations
    iseed=12345                    # Random seed
)
```

### Key Methods

```python
# Training
model.fit(X, y)                    # Train the model

# Prediction
predictions = model.predict(X)      # Class predictions
oob_error = model.oob_error()       # Out-of-bag error rate
confusion = model.confusion_matrix() # Confusion matrix

# Importance
importance = model.overall_importance()      # Overall feature importance
local_imp = model.local_importance()        # Local importance (n×m matrix)

# Proximity
proximity = model.get_proximity_matrix()    # Full proximity matrix (CPU)
A, B = model.get_lowrank_factors()          # Low-rank factors (GPU QLORA)
mds_coords = model.compute_mds_3d_from_factors()  # 3D MDS from factors

# Visualization
rf.rfviz(model, X, y, output_file="rfviz.html")  # Interactive visualization
```

## 🎓 Methodology

RFX strictly follows Breiman and Cutler's original Random Forest algorithms:

### Classification
- **Gini impurity** for split selection
- **Bootstrap sampling** with replacement
- **Random feature selection** (mtry features per split)
- **Majority voting** for final predictions

### Out-of-Bag (OOB) Error
Each tree is trained on a bootstrap sample, leaving ~37% of samples out-of-bag. OOB error is computed using only trees where each sample was OOB, providing an unbiased estimate of generalization error without a separate test set.

### Importance Measures
- **Overall importance**: Aggregates impurity reduction across all trees
- **Local importance**: Permutes features per-sample and measures prediction change

### Proximity Matrices
Proximity between samples $i$ and $j$ is the fraction of trees where they fall into the same terminal node:
$$p(i,j) = \frac{1}{B} \sum_{b=1}^{B} \mathbb{I}(\text{node}_b(i) = \text{node}_b(j))$$

### Case-wise vs. Non-case-wise
- **Non-case-wise**: Standard Random Forest (equal weight to all samples)
- **Case-wise**: Weighted by bootstrap frequency (from unreleased Fortran extensions)

## 🔬 Technical Details

### QLORA Compression
QLORA (Quantized Low-Rank Adaptation) reduces proximity matrix memory by storing low-rank factors $A \in \mathbb{R}^{n \times r}$ and $B \in \mathbb{R}^{n \times r}$ instead of the full $n \times n$ matrix. The proximity is reconstructed as $P \approx AB^T$. With INT8 quantization and rank-32, this achieves 12,500× compression (80GB → 6.4MB) while maintaining 99% geometric structure preservation (measured via MDS correlation).

### CPU TriBlock Proximity
Combines two optimizations:
1. **Upper-triangle storage**: Exploits symmetry ($P_{ij} = P_{ji}$)
2. **Block-sparse thresholding**: Zeros out blocks below threshold $\tau=0.0001$

This achieves 2.7× memory reduction with **lossless quality** (MDS correlation = 1.00) on small datasets, and estimated $\rho \approx 0.98$-0.99 on large datasets.

### SM-Aware GPU Batching
Automatically selects optimal batch size based on:
- **GPU Streaming Multiprocessor (SM) count**: Targets 2×SM concurrent blocks
- **Available GPU memory**: Ensures sufficient memory headroom
- **Total tree count**: Balances parallelism vs. overhead

For example, on RTX 3060 (28 SMs, 12GB VRAM) with 500 trees, auto-scaling selects batch_size=100, achieving 95% SM utilization.

### GPU vs. CPU Random Number Generation
RFX uses different RNGs for CPU and GPU:
- **CPU**: MT19937 (Mersenne Twister)
- **GPU**: cuRAND (NVIDIA's GPU RNG)

This means CPU and GPU will produce **different tree structures** and **different importance rankings** even with the same seed. This is expected and both implementations are statistically valid. **Do not compare importance values across platforms**—focus on relative rankings and predictive performance within each platform.

## 📖 Citation

If you use RFX in your research, please cite:

```bibtex
@article{rfx2025,
  title={RFX: High-Performance Random Forests with GPU Acceleration and QLORA Compression},
  author={Kuchar, Chris},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## 🙏 Acknowledgments

This work aims to honor the legacy of **Leo Breiman** and **Adele Cutler** by faithfully implementing their Random Forest methodology and ensuring it is not forgotten by the research community. The implementation strictly follows all algorithms from the original Fortran code with no shortcuts on methodology whatsoever.

RFX is built on their original vision of a comprehensive Random Forest methodology, extending it with modern GPU acceleration and memory-efficient proximity computation to enable large-scale analysis.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 References

- Breiman, L. (2001). Random forests. *Machine learning*, 45(1), 5-32.
- Breiman, L., & Cutler, A. (2004). Random forests. *Manual for R Package*. Available at https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
- Dettmers, T., et al. (2023). QLoRA: Efficient finetuning of quantized LLMs. *Advances in Neural Information Processing Systems*, 36.
- Beckett, C. (2018). Rfviz: An interactive visualization package for Random Forests in R. *All Graduate Plan B and other Reports*, 1335.

## 🐛 Issues and Contributions

- **Bug Reports**: https://github.com/chriskuchar/RFX/issues
- **Source Code**: https://github.com/chriskuchar/RFX

## 📝 Version History

- **v1.0.0** (2025): Initial release
  - Complete classification implementation
  - GPU acceleration with SM-aware batching
  - QLORA proximity compression
  - CPU TriBlock proximity
  - Case-wise analysis
  - Interactive rfviz visualization

**Planned for v2.0:**
- Regression support
- Unsupervised learning
- CLIQUE importance
- RF-GAP proximity
- R version

---

**RFX: Random Forests X** - Where X represents compression/quantization, enabling Random Forest analysis at unprecedented scale.
