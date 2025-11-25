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
    nsample=None,              # Set automatically during fit
    ntree=100,
    mdim=None,                 # Set automatically during fit
    nclass=None,               # Set automatically during fit
    maxcat=10,
    mtry=0,
    maxnode=0,
    minndsize=1,
    ncsplit=25,                # Max categorical splits per node
    ncmax=25,                  # Max categories to consider
    iseed=12345,
    compute_proximity=False,
    compute_importance=True,
    compute_local_importance=False,
    use_gpu=False,
    use_qlora=False,
    quant_mode="nf4",
    use_sparse=False,
    sparsity_threshold=1e-6,
    batch_size=0,
    nodesize=5,
    cutoff=0.01,
    show_progress=True,
    progress_desc="Training Random Forest",
    gpu_loss_function="gini",
    rank=32,
    n_threads_cpu=0,
    use_casewise=False
)
```

### Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `nsample` | int | 1000 | Number of samples (set automatically from data during fit) |
| `ntree` | int | 100 | Number of trees in the forest |
| `mdim` | int | - | Number of features (set automatically from data during fit) |
| `nclass` | int | 2 | Number of classes (set automatically from data during fit) |
| `maxcat` | int | 10 | Maximum categories for categorical variables |
| `mtry` | int | 0 | Features to consider at each split. 0 = auto (sqrt(n_features)) |
| `maxnode` | int | 0 | Maximum nodes per tree. 0 = unlimited |
| `minndsize` | int | 1 | Minimum node size for splitting |
| `ncsplit` | int | 25 | Maximum categorical splits per node |
| `ncmax` | int | 25 | Maximum categories to consider for splitting |
| `iseed` | int | 12345 | Random seed for reproducibility |
| `compute_proximity` | bool | False | Compute sample proximity matrix |
| `compute_importance` | bool | True | Compute overall feature importance |
| `compute_local_importance` | bool | False | Compute per-sample feature importance |
| `use_gpu` | bool | False | Enable CUDA GPU acceleration |
| `use_qlora` | bool | False | Enable QLoRA low-rank proximity compression |
| `quant_mode` | str | "nf4" | Quantization mode: "int8", "nf4", "fp16", "fp32" |
| `use_sparse` | bool | False | Enable CPU block-sparse (TriBlock) proximity |
| `sparsity_threshold` | float | 1e-6 | Block-sparse threshold for CPU proximity |
| `batch_size` | int | 0 | GPU batch size. 0 = auto |
| `nodesize` | int | 5 | Minimum terminal node size |
| `cutoff` | float | 0.01 | Cutoff threshold for node splitting |
| `show_progress` | bool | True | Show training progress bar |
| `progress_desc` | str | "Training Random Forest" | Progress bar description |
| `gpu_loss_function` | str | "gini" | GPU loss function ("gini" for classification) |
| `rank` | int | 32 | Low-rank dimension for QLoRA compression |
| `n_threads_cpu` | int | 0 | CPU threads. 0 = auto |
| `use_casewise` | bool | False | Use case-wise (bootstrap frequency) weighting |

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

#### predict_proba(X)

Predict class probabilities for samples.

```python
probabilities = model.predict_proba(X)
# probabilities[i, j] = probability that sample i belongs to class j
```

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `X` | array-like | Features, shape (n_samples, n_features) |

**Returns:** ndarray of shape (n_samples, n_classes) with class probabilities

**Note:** Probabilities are computed as the fraction of trees voting for each class.

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

#### compute_proximity_matrix()

Alias for `get_proximity_matrix()` (for compatibility).

```python
prox = model.compute_proximity_matrix()
```

**Returns:** ndarray of shape (n_samples, n_samples)

---

#### compute_mds_3d_from_factors()

Compute 3D MDS coordinates directly from low-rank factors.

```python
mds = model.compute_mds_3d_from_factors()
# mds[i, :] = [x, y, z] coordinates for sample i
```

**Returns:** ndarray of shape (n_samples, 3)

**Note:** Only available when `use_qlora=True` and `compute_proximity=True`.

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

#### compute_mds_3d_cpu()

Compute 3D MDS coordinates from full proximity matrix (CPU implementation).

```python
mds = model.compute_mds_3d_cpu()
# mds[i, :] = [x, y, z] coordinates for sample i
```

**Returns:** ndarray of shape (n_samples, 3)

**Note:** Requires `compute_proximity=True` and works with CPU proximity matrices. This is a fast C++ implementation of metric MDS.

---

#### compute_mds_from_factors(k=3)

Compute k-dimensional MDS coordinates from low-rank factors.

```python
mds = model.compute_mds_from_factors(k=3)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `k` | int | 3 | Number of MDS dimensions (2, 3, or more) |

**Returns:** ndarray of shape (n_samples, k)

**Note:** Only available when `use_qlora=True`.

---

#### get_n_samples()

Get number of training samples.

```python
n = model.get_n_samples()
```

**Returns:** int

---

#### get_n_features()

Get number of features.

```python
n = model.get_n_features()
```

**Returns:** int

---

#### get_n_classes()

Get number of classes.

```python
n = model.get_n_classes()
```

**Returns:** int

---

#### get_task_type()

Get the task type (classification/regression/unsupervised).

```python
task = model.get_task_type()
# Returns: "classification", "regression", or "unsupervised"
```

**Returns:** str

---

#### cleanup()

Explicit cleanup of GPU memory for this model.

```python
model.cleanup()
```

**Returns:** None

**Note:** Useful when you're done with a model and want to free GPU memory immediately without waiting for Python garbage collection.

---

#### get_oob_predictions()

Get out-of-bag predictions for training samples.

```python
oob_pred = model.get_oob_predictions()
```

**Returns:** ndarray of shape (n_samples,) with OOB predicted class labels

**Note:** Only available for training samples after `fit()` is called.

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

### confusion_matrix(y_true, y_pred)

Compute confusion matrix (module-level function).

```python
cm = rf.confusion_matrix(y_true, y_pred)
```

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `y_true` | array-like | True labels |
| `y_pred` | array-like | Predicted labels |

**Returns:** ndarray of shape (n_classes, n_classes)

---

### classification_report(y_true, y_pred)

Generate classification report with precision, recall, F1-score (module-level function).

```python
report = rf.classification_report(y_val, y_pred)
print(report)
```

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `y_true` | array-like | True labels |
| `y_pred` | array-like | Predicted labels |

**Returns:** str, formatted classification report

---

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

### check_gpu_memory(size_mb)

Check if sufficient GPU memory is available.

```python
if rf.check_gpu_memory(1000):  # Check for 1GB
    print("Sufficient memory available")
```

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `size_mb` | int | Required memory in MB |

**Returns:** bool

---

### print_gpu_memory_status()

Print current GPU memory status.

```python
rf.print_gpu_memory_status()
```

**Returns:** None

**Output:** Prints total, used, and free GPU memory.

---

### gpu_cleanup()

Force GPU cleanup and memory release.

```python
rf.gpu_cleanup()
```

**Returns:** None

**Note:** More aggressive than `clear_gpu_cache()`. Use when experiencing memory issues.

---

### reset_cuda_device()

Reset CUDA device (nuclear option for memory issues).

```python
rf.reset_cuda_device()
```

**Returns:** None

**Warning:** This will invalidate all GPU allocations. Only use as last resort.

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

### load_iris()

Load the UCI Iris dataset.

```python
X, y = rf.load_iris()
print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
print(f"Classes: {len(np.unique(y))}")
```

**Returns:** tuple (X, y)
- `X`: ndarray of shape (150, 4)
- `y`: ndarray of shape (150,) with labels 0, 1, 2

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
mds = model.compute_mds_3d_from_factors()

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
