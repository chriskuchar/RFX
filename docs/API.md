# RFX API Reference

Complete API documentation for RFX: GPU-Accelerated Random Forests with QLoRA Compression.

---

## Table of Contents

- [RandomForest](#randomforest)
- [Visualization](#visualization)
- [Utility Functions](#utility-functions)
- [Data Loading](#data-loading)

---

## RandomForest

```python
rf.RandomForestClassifier(
    ntree=100,
    mtry=0,
    nsample=1000,
    nclass=2,
    maxcat=10,
    maxnode=0,
    minndsize=1,
    nodesize=5,
    iseed=12345,
    compute_proximity=False,
    compute_importance=True,
    compute_local_importance=False,
    use_gpu=False,
    use_qlora=False,
    quant_mode="nf4",
    rank=32,
    batch_size=0,
    use_casewise=False,
    use_rfgap=False,
    n_threads_cpu=0,
    show_progress=True
)
```

### Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `ntree` | int | 100 | Number of trees in the forest |
| `mtry` | int | 0 | Features to consider at each split. 0 = auto (sqrt(n_features)) |
| `nsample` | int | 1000 | Number of samples (set automatically from data) |
| `nclass` | int | 2 | Number of classes (set automatically from data) |
| `maxcat` | int | 10 | Maximum categories for categorical variables |
| `maxnode` | int | 0 | Maximum nodes per tree. 0 = unlimited |
| `minndsize` | int | 1 | Minimum node size for splitting |
| `nodesize` | int | 5 | Minimum terminal node size |
| `iseed` | int | 12345 | Random seed for reproducibility |
| `compute_proximity` | bool | False | Compute sample proximity matrix |
| `compute_importance` | bool | True | Compute overall feature importance |
| `compute_local_importance` | bool | False | Compute per-sample feature importance |
| `use_gpu` | bool | False | Enable CUDA GPU acceleration |
| `use_qlora` | bool | False | Enable QLoRA low-rank proximity compression |
| `quant_mode` | str | "nf4" | Quantization mode: "int8", "nf4", "fp16", "fp32" |
| `rank` | int | 32 | Low-rank dimension for QLoRA compression |
| `batch_size` | int | 0 | GPU batch size. 0 = auto |
| `use_casewise` | bool | False | Use case-wise (bootstrap frequency) weighting |
| `use_rfgap` | bool | False | Use RF-GAP proximity normalization |
| `n_threads_cpu` | int | 0 | CPU threads. 0 = auto |
| `show_progress` | bool | True | Show training progress bar |

### Methods

#### fit(X, y)

Train the random forest.

```python
model.fit(X, y)
```

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `X` | array-like | Training features, shape (n_samples, n_features) |
| `y` | array-like | Target labels, shape (n_samples,) |

**Returns:** self

---

#### predict(X)

Predict class labels for samples.

```python
predictions = model.predict(X)
```

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `X` | array-like | Features, shape (n_samples, n_features) |

**Returns:** ndarray of shape (n_samples,) with predicted class labels

---

#### get_oob_error()

Get out-of-bag error rate.

```python
error = model.get_oob_error()
print(f"OOB Error: {error:.2%}")
```

**Returns:** float, OOB error rate (0.0 to 1.0)

---

#### feature_importances_()

Get overall feature importance scores (mean decrease in impurity).

```python
importance = model.feature_importances_()
top_features = np.argsort(importance)[::-1][:5]
```

**Returns:** ndarray of shape (n_features,) with importance scores

---

#### get_local_importance()

Get per-sample feature importance matrix.

```python
local_imp = model.get_local_importance()
# local_imp[i, j] = importance of feature j for sample i
```

**Returns:** ndarray of shape (n_samples, n_features)

**Note:** Requires `compute_local_importance=True` during training.

---

#### get_proximity_matrix()

Get full proximity matrix (CPU only, not for QLoRA).

```python
prox = model.get_proximity_matrix()
# prox[i, j] = similarity between samples i and j
```

**Returns:** ndarray of shape (n_samples, n_samples)

**Note:** For QLoRA models, use `compute_mds_from_factors()` instead.

---

#### compute_mds_from_factors(k=3)

Compute MDS coordinates directly from low-rank factors.

```python
mds = model.compute_mds_from_factors(k=3)
# mds[i, :] = [x, y, z] coordinates for sample i
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `k` | int | 3 | Number of MDS dimensions |

**Returns:** ndarray of shape (n_samples, k)

**Note:** Only available when `use_qlora=True`.

---

#### get_lowrank_factors()

Get low-rank proximity factors A and B where P ≈ A @ B.T.

```python
A, B, rank = model.get_lowrank_factors()
# Reconstruct: proximity = A @ B.T
```

**Returns:** tuple (A, B, rank)
- `A`: ndarray of shape (n_samples, rank)
- `B`: ndarray of shape (n_samples, rank)
- `rank`: int, actual rank used

---

#### confusion_matrix(y_true, y_pred)

Compute confusion matrix.

```python
cm = model.confusion_matrix(y_true, y_pred)
```

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `y_true` | array-like | True labels |
| `y_pred` | array-like | Predicted labels |

**Returns:** ndarray of shape (n_classes, n_classes)

---

#### classification_report(y_true, y_pred)

Generate classification report with precision, recall, F1-score.

```python
report = model.classification_report(y_true, y_pred)
print(report)
```

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `y_true` | array-like | True labels |
| `y_pred` | array-like | Predicted labels |

**Returns:** str, formatted classification report

---

## Visualization

### rfviz()

Generate interactive RFViz visualization with linked brushing.

```python
rf.rfviz(
    rf_model,
    X,
    y,
    feature_names=None,
    class_names=None,
    n_clusters=3,
    title="RFViz",
    output_file="rfviz.html",
    show_in_browser=True,
    save_html=True,
    mds_k=3
)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `rf_model` | RandomForest | required | Trained model with proximity |
| `X` | array-like | required | Feature matrix |
| `y` | array-like | required | Labels |
| `feature_names` | list | None | Feature names for display |
| `class_names` | list | None | Class names for display |
| `n_clusters` | int | 3 | K-means clusters for coloring |
| `title` | str | "RFViz" | Plot title |
| `output_file` | str | "rfviz.html" | Output HTML file path |
| `show_in_browser` | bool | True | Open in browser after generation |
| `save_html` | bool | True | Save to file |
| `mds_k` | int | 3 | MDS dimensions (2 or 3) |

**Returns:** Plotly Figure object (if `save_html=False`)

**Features:**
- 2x2 dashboard layout
- Input features parallel coordinates
- Local importance parallel coordinates  
- 3D MDS proximity plot
- Class votes heatmap
- Linked brushing across all plots
- Save selected points to CSV

---

## Utility Functions

### cuda_is_available()

Check if CUDA GPU is available.

```python
if rf.cuda_is_available():
    print("GPU acceleration available")
```

**Returns:** bool

---

### get_gpu_memory_info()

Get GPU memory information.

```python
info = rf.get_gpu_memory_info()
print(f"Total: {info['total'] / 1e9:.1f} GB")
print(f"Used: {info['used'] / 1e9:.1f} GB")
print(f"Free: {info['free'] / 1e9:.1f} GB")
```

**Returns:** dict with keys: `total`, `used`, `free` (in bytes)

---

### clear_gpu_cache()

Clear GPU memory cache.

```python
rf.clear_gpu_cache()
```

**Returns:** None

**Note:** Useful in Jupyter notebooks to free memory between experiments.

---

## Data Loading

### load_wine()

Load the UCI Wine dataset.

```python
X, y = rf.load_wine()
print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
print(f"Classes: {len(np.unique(y))}")
```

**Returns:** tuple (X, y)
- `X`: ndarray of shape (178, 13)
- `y`: ndarray of shape (178,) with labels 0, 1, 2

---

## Examples

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
importance = model.feature_importances_()
```

### QLoRA Proximity with Visualization

```python
import RFX as rf

X, y = rf.load_wine()

model = rf.RandomForestClassifier(
    ntree=100,
    use_gpu=True,
    compute_proximity=True,
    use_qlora=True,
    quant_mode="int8",
    rank=32
)
model.fit(X, y)

# Get MDS coordinates
mds = model.compute_mds_from_factors(k=3)

# Interactive visualization
rf.rfviz(model, X, y, output_file="wine_rfviz.html")
```

### Local Importance Analysis

```python
import RFX as rf
import numpy as np

X, y = rf.load_wine()

model = rf.RandomForestClassifier(
    ntree=100,
    use_gpu=True,
    compute_importance=True,
    compute_local_importance=True
)
model.fit(X, y)

# Overall importance
overall = model.feature_importances_()

# Per-sample importance
local = model.get_local_importance()

# Find most important feature for each sample
most_important = np.argmax(local, axis=1)
```

---

## Quantization Modes

| Mode | Bits | Memory | Precision | Use Case |
|:-----|:-----|:-------|:----------|:---------|
| `fp32` | 32 | 1x | Highest | Debugging |
| `fp16` | 16 | 2x reduction | High | Default |
| `int8` | 8 | 4x reduction | Medium | Large datasets |
| `nf4` | 4 | 8x reduction | Lower | Very large datasets |

**Recommendation:** Start with `int8` for most use cases. Use `nf4` only for very large datasets (>100K samples) where memory is critical.

---

## Performance Tips

1. **GPU Batch Size:** Let auto-tuning handle it (`batch_size=0`) unless you have specific memory constraints.

2. **Rank Selection:** 
   - `rank=32` is good for visualization (3D MDS)
   - `rank=100+` for more accurate proximity reconstruction
   - Higher rank = more memory, more accurate

3. **Tree Count for MDS:**
   - 100+ trees recommended for stable MDS coordinates
   - Fewer trees = more duplicate MDS points (sparse OOB coverage)

4. **Memory Management in Jupyter:**
   ```python
   rf.clear_gpu_cache()  # Between experiments
   ```

5. **Large Datasets:**
   ```python
   model = rf.RandomForestClassifier(
       ntree=500,
       use_gpu=True,
       compute_proximity=True,
       use_qlora=True,
       quant_mode="int8",  # or "nf4" for very large
       rank=32
   )
   ```
