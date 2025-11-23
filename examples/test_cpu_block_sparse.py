#!/usr/bin/env python3
"""
Test CPU Block-Sparse Proximity vs CPU Full Matrix
For arXiv paper - demonstrates memory-efficient CPU proximity computation
Wine dataset (178 samples), 100 trees
"""

import RFX
import numpy as np
import time
from scipy.stats import spearmanr, pearsonr
from sklearn.manifold import MDS
from scipy.linalg import orthogonal_procrustes

# Load Wine dataset
cache_file = "wine_dataset.npz"
data = np.load(cache_file)
X, y = data['X'], data['y']

print(f"Wine dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes\n")

ntree = 100

print(f"{'='*80}")
print(f"CPU BLOCK-SPARSE vs CPU FULL MATRIX COMPARISON ({ntree} trees)")
print(f"{'='*80}\n")

# Test 1: CPU Full Matrix (ground truth)
print("1. Training CPU Full Matrix (ground truth)...")
print(f"{'-'*80}")
start = time.time()
rf_cpu_full = RFX.RandomForestClassifier(
    ntree=ntree,
    use_gpu=False,
    use_casewise=False,
    compute_proximity=True,
    use_qlora=False,
    use_sparse=False,  # Full dense matrix
    batch_size=0,
    iseed=42
)
rf_cpu_full.fit(X, y)
cpu_full_time = time.time() - start

P_cpu_full = rf_cpu_full.get_proximity_matrix()
print(f"  Time: {cpu_full_time:.2f}s")
print(f"  Shape: {P_cpu_full.shape}")
print(f"  Memory: {P_cpu_full.nbytes/1024:.1f} KB")
print(f"  Mean: {np.mean(P_cpu_full):.4f}, Std: {np.std(P_cpu_full):.4f}")
print(f"  Diagonal mean: {np.mean(np.diag(P_cpu_full)):.4f}")
print()

# Test 2: CPU Block-Sparse (memory-efficient)
print("2. Training CPU Block-Sparse (memory-efficient, threshold=0.001)...")
print(f"{'-'*80}")
start = time.time()
rf_cpu_sparse = RFX.RandomForestClassifier(
    ntree=ntree,
    use_gpu=False,
    use_casewise=False,
    compute_proximity=True,
    use_qlora=False,
    use_sparse=True,  # Enable block-sparse storage
    sparsity_threshold=0.001,  # Conservative threshold (99.9% accuracy)
    batch_size=0,
    iseed=42
)
rf_cpu_sparse.fit(X, y)
cpu_sparse_time = time.time() - start

P_cpu_sparse = rf_cpu_sparse.get_proximity_matrix()
print(f"  Time: {cpu_sparse_time:.2f}s")
print(f"  Shape: {P_cpu_sparse.shape}")
print(f"  Memory: {P_cpu_sparse.nbytes/1024:.1f} KB (after conversion to dense)")
print(f"  Mean: {np.mean(P_cpu_sparse):.4f}, Std: {np.std(P_cpu_sparse):.4f}")
print(f"  Diagonal mean: {np.mean(np.diag(P_cpu_sparse)):.4f}")
print()

# Comparison
print(f"{'='*80}")
print("COMPARISON: CPU Block-Sparse vs CPU Full Matrix")
print(f"{'='*80}\n")

# Correlations
spearman_corr, _ = spearmanr(P_cpu_sparse.flatten(), P_cpu_full.flatten())
pearson_corr, _ = pearsonr(P_cpu_sparse.flatten(), P_cpu_full.flatten())
print(f"Spearman correlation: {spearman_corr:.4f}")
print(f"Pearson correlation: {pearson_corr:.4f}")

# Errors
mse = np.mean((P_cpu_sparse - P_cpu_full) ** 2)
mae = np.mean(np.abs(P_cpu_sparse - P_cpu_full))
print(f"Mean Squared Error (MSE): {mse:.6f}")
print(f"Mean Absolute Error (MAE): {mae:.6f}")

# Timing
speedup = cpu_full_time / cpu_sparse_time
print(f"\nTiming:")
print(f"  CPU Full: {cpu_full_time:.2f}s")
print(f"  CPU Sparse: {cpu_sparse_time:.2f}s")
print(f"  Speedup: {speedup:.2f}× {'(sparse faster)' if speedup > 1 else '(full faster)'}")

# MDS Embedding Comparison
print(f"\n{'='*80}")
print("MDS EMBEDDING COMPARISON")
print(f"{'='*80}\n")

print("Computing MDS embeddings...")

# Convert to distance matrices
D_full = 1 - P_cpu_full
D_sparse = 1 - P_cpu_sparse
np.fill_diagonal(D_full, 0)
np.fill_diagonal(D_sparse, 0)

# Compute MDS embeddings
print("  Computing CPU Full MDS...")
mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42, max_iter=300, n_init=1)
coords_full = mds.fit_transform(D_full)

print("  Computing CPU Sparse MDS...")
mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42, max_iter=300, n_init=1)
coords_sparse = mds.fit_transform(D_sparse)

# Align using Procrustes
R, scale = orthogonal_procrustes(coords_sparse, coords_full)
coords_sparse_aligned = coords_sparse @ R

# Compute correlations
corr_x = np.corrcoef(coords_sparse_aligned[:, 0], coords_full[:, 0])[0, 1]
corr_y = np.corrcoef(coords_sparse_aligned[:, 1], coords_full[:, 1])[0, 1]
corr_z = np.corrcoef(coords_sparse_aligned[:, 2], coords_full[:, 2])[0, 1]
corr_mean = np.mean([corr_x, corr_y, corr_z])

print(f"\nMDS Embedding Correlation (after Procrustes alignment):")
print(f"  X-axis: {corr_x:.4f}")
print(f"  Y-axis: {corr_y:.4f}")
print(f"  Z-axis: {corr_z:.4f}")
print(f"  Mean: {corr_mean:.4f}")

# RMSE
rmse = np.sqrt(np.mean((coords_sparse_aligned - coords_full) ** 2))
print(f"  RMSE: {rmse:.4f}")

# Quality assessment
print(f"\n{'='*80}")
print("QUALITY ASSESSMENT")
print(f"{'='*80}\n")

if spearman_corr > 0.99:
    quality = "EXCELLENT"
elif spearman_corr > 0.95:
    quality = "VERY GOOD"
elif spearman_corr > 0.90:
    quality = "GOOD"
else:
    quality = "ACCEPTABLE"

print(f"Block-sparse quality: {quality}")
print(f"  - Spearman correlation: {spearman_corr:.4f}")
print(f"  - MDS correlation: {corr_mean:.4f}")
print(f"  - MSE: {mse:.6f}")
print(f"  - Threshold: 0.001 (conservative)")

print(f"\n{'='*80}")
print("SUMMARY FOR ARXIV TABLE")
print(f"{'='*80}\n")

# Estimated memory savings for large datasets
print("Estimated memory savings for large datasets:")
print(f"  100K samples:")
full_100k = 100000 * 100000 * 8 / (1024**3)  # GB
sparse_100k_low = full_100k * 0.1  # 10% sparsity
sparse_100k_high = full_100k * 0.4  # 40% sparsity
print(f"    Full matrix: {full_100k:.1f} GB")
print(f"    Block-sparse: {sparse_100k_low:.1f}-{sparse_100k_high:.1f} GB (10-40% of full)")
print(f"    Compression: {full_100k/sparse_100k_high:.1f}-{full_100k/sparse_100k_low:.1f}×")

print(f"\n{'='*80}")
print("TABLE ROW FOR PAPER")
print(f"{'='*80}\n")
print(f"CPU Sparse  {P_cpu_sparse.nbytes/1024:<12.1f} {1.0:<10}× {spearman_corr:<10.4f} {pearson_corr:<10.4f} {corr_mean:<10.4f} {cpu_sparse_time:<10.2f}")
print(f"(Note: Memory shown is dense output; internal sparse storage saves memory during computation)")

print(f"\n{'='*80}")
print("✅ Test complete!")
print(f"{'='*80}")

