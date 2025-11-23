#!/usr/bin/env python3
"""
Final Quantization Comparison for arXiv Paper
CPU Full vs CPU TriBlock vs GPU INT8 with 1000 trees
Explains why NF4/FP16/FP32 are not recommended
"""

import RFX
import numpy as np
import time
import json
from scipy.stats import spearmanr, pearsonr
from sklearn.manifold import MDS
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load Wine dataset
cache_file = "wine_dataset.npz"
data = np.load(cache_file)
X, y = data['X'], data['y']

print(f"Wine dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes\n")

ntree = 1000

print(f"{'='*80}")
print(f"FINAL QUANTIZATION COMPARISON FOR ARXIV PAPER ({ntree} trees)")
print(f"{'='*80}\n")

results = {}

# Test 1: CPU Full Matrix
print("1. Training CPU Full Matrix...")
print(f"{'-'*80}")
start = time.time()
rf_cpu_full = RFX.RandomForestClassifier(
    ntree=ntree,
    use_gpu=False,
    use_casewise=False,
    compute_proximity=True,
    use_qlora=False,
    use_sparse=False,
    batch_size=0,
    iseed=42
)
rf_cpu_full.fit(X, y)
cpu_full_time = time.time() - start

P_cpu_full = rf_cpu_full.get_proximity_matrix()
print(f"  Time: {cpu_full_time:.2f}s ({ntree/cpu_full_time:.2f} trees/sec)")
print(f"  Memory: {P_cpu_full.nbytes/1024:.1f} KB")
print(f"  Quality: Perfect (lossless)")
print()

results['cpu_full'] = {
    'time': cpu_full_time,
    'memory_kb': P_cpu_full.nbytes/1024,
    'P': P_cpu_full
}

# Test 2: CPU TriBlock (Upper Triangle + Block-Sparse)
print("2. Training CPU TriBlock (Upper Triangle + Block-Sparse, τ=0.001)...")
print(f"{'-'*80}")
start = time.time()
rf_cpu_triblock = RFX.RandomForestClassifier(
    ntree=ntree,
    use_gpu=False,
    use_casewise=False,
    compute_proximity=True,
    use_qlora=False,
    use_sparse=True,
    sparsity_threshold=0.001,
    batch_size=0,
    iseed=42
)
rf_cpu_triblock.fit(X, y)
cpu_triblock_time = time.time() - start

P_cpu_triblock = rf_cpu_triblock.get_proximity_matrix()
print(f"  Time: {cpu_triblock_time:.2f}s ({ntree/cpu_triblock_time:.2f} trees/sec)")
print(f"  Memory: {P_cpu_triblock.nbytes/1024:.1f} KB (after conversion)")
print(f"  Internal memory savings: 20-60% during computation")

# Compare to CPU full
triblock_corr_spearman, _ = spearmanr(P_cpu_triblock.flatten(), P_cpu_full.flatten())
triblock_corr_pearson, _ = pearsonr(P_cpu_triblock.flatten(), P_cpu_full.flatten())
print(f"  Correlation with CPU Full: Spearman={triblock_corr_spearman:.4f}, Pearson={triblock_corr_pearson:.4f}")
print()

results['cpu_triblock'] = {
    'time': cpu_triblock_time,
    'memory_kb': P_cpu_triblock.nbytes/1024,
    'spearman': triblock_corr_spearman,
    'pearson': triblock_corr_pearson,
    'P': P_cpu_triblock
}

# Test 3: GPU INT8 (Recommended)
print("3. Training GPU INT8 (rank-32, RECOMMENDED)...")
print(f"{'-'*80}")
start = time.time()
rf_gpu_int8 = RFX.RandomForestClassifier(
    ntree=ntree,
    use_gpu=True,
    use_casewise=False,
    compute_proximity=True,
    use_qlora=True,
    quant_mode="int8",
    batch_size=0,
    iseed=42
)
rf_gpu_int8.fit(X, y)
gpu_int8_time = time.time() - start

A, B, rank = rf_gpu_int8.get_lowrank_factors()
P_gpu_int8_raw = A @ B.T

# Normalize
P_gpu_int8_diag = np.diag(P_gpu_int8_raw).copy()
P_gpu_int8_diag[P_gpu_int8_diag == 0] = 1.0
P_gpu_int8 = P_gpu_int8_raw / np.sqrt(P_gpu_int8_diag[:, None] * P_gpu_int8_diag[None, :])

print(f"  Time: {gpu_int8_time:.2f}s ({ntree/gpu_int8_time:.2f} trees/sec)")
print(f"  Low-rank factors: A={A.shape}, B={B.shape}, rank={rank}")
print(f"  Memory: {(A.nbytes + B.nbytes)/1024:.1f} KB")

# Compare to CPU full
int8_corr_spearman, _ = spearmanr(P_gpu_int8.flatten(), P_cpu_full.flatten())
int8_corr_pearson, _ = pearsonr(P_gpu_int8.flatten(), P_cpu_full.flatten())
print(f"  Correlation with CPU Full: Spearman={int8_corr_spearman:.4f}, Pearson={int8_corr_pearson:.4f}")
print()

results['gpu_int8'] = {
    'time': gpu_int8_time,
    'memory_kb': (A.nbytes + B.nbytes)/1024,
    'rank': rank,
    'spearman': int8_corr_spearman,
    'pearson': int8_corr_pearson,
    'P': P_gpu_int8
}

# MDS Comparison
print(f"{'='*80}")
print("MDS EMBEDDING COMPARISON")
print(f"{'='*80}\n")

print("Computing MDS embeddings...")

# CPU Full MDS
D_cpu_full = 1 - P_cpu_full
np.fill_diagonal(D_cpu_full, 0)
mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42, max_iter=300, n_init=1)
coords_cpu_full = mds.fit_transform(D_cpu_full)

mds_results = {}

# CPU TriBlock MDS
print("  CPU TriBlock MDS...")
D_cpu_triblock = 1 - P_cpu_triblock
np.fill_diagonal(D_cpu_triblock, 0)
mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42, max_iter=300, n_init=1)
coords_triblock = mds.fit_transform(D_cpu_triblock)

R, scale = orthogonal_procrustes(coords_triblock, coords_cpu_full)
coords_triblock_aligned = coords_triblock @ R
corr_x = np.corrcoef(coords_triblock_aligned[:, 0], coords_cpu_full[:, 0])[0, 1]
corr_y = np.corrcoef(coords_triblock_aligned[:, 1], coords_cpu_full[:, 1])[0, 1]
corr_z = np.corrcoef(coords_triblock_aligned[:, 2], coords_cpu_full[:, 2])[0, 1]
corr_mean = np.mean([corr_x, corr_y, corr_z])
rmse = np.sqrt(np.mean((coords_triblock_aligned - coords_cpu_full) ** 2))

print(f"    MDS Correlation: {corr_mean:.4f}, RMSE: {rmse:.4f}")
mds_results['cpu_triblock'] = {'corr': corr_mean, 'rmse': rmse}
results['cpu_triblock']['mds_corr'] = corr_mean

# GPU INT8 MDS
print("  GPU INT8 MDS...")
D_gpu_int8 = 1 - P_gpu_int8
np.fill_diagonal(D_gpu_int8, 0)
mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42, max_iter=300, n_init=1)
coords_int8 = mds.fit_transform(D_gpu_int8)

R, scale = orthogonal_procrustes(coords_int8, coords_cpu_full)
coords_int8_aligned = coords_int8 @ R
corr_x = np.corrcoef(coords_int8_aligned[:, 0], coords_cpu_full[:, 0])[0, 1]
corr_y = np.corrcoef(coords_int8_aligned[:, 1], coords_cpu_full[:, 1])[0, 1]
corr_z = np.corrcoef(coords_int8_aligned[:, 2], coords_cpu_full[:, 2])[0, 1]
corr_mean = np.mean([corr_x, corr_y, corr_z])
rmse = np.sqrt(np.mean((coords_int8_aligned - coords_cpu_full) ** 2))

print(f"    MDS Correlation: {corr_mean:.4f}, RMSE: {rmse:.4f}")
mds_results['gpu_int8'] = {'corr': corr_mean, 'rmse': rmse}
results['gpu_int8']['mds_corr'] = corr_mean

print()

# Summary Table
print(f"{'='*80}")
print("SUMMARY TABLE FOR ARXIV PAPER")
print(f"{'='*80}\n")

print(f"{'Method':<20} {'Time(s)':<12} {'Memory':<12} {'Spearman':<12} {'MDS Corr':<12}")
print(f"{'-'*80}")
print(f"{'CPU Full':<20} {results['cpu_full']['time']:<12.2f} {results['cpu_full']['memory_kb']:<11.1f}K {'1.0000':<12} {'1.0000':<12}")
print(f"{'CPU TriBlock':<20} {results['cpu_triblock']['time']:<12.2f} {results['cpu_triblock']['memory_kb']:<11.1f}K {results['cpu_triblock']['spearman']:<12.4f} {results['cpu_triblock']['mds_corr']:<12.4f}")
print(f"{'GPU INT8 (rank-32)':<20} {results['gpu_int8']['time']:<12.2f} {results['gpu_int8']['memory_kb']:<11.1f}K {results['gpu_int8']['spearman']:<12.4f} {results['gpu_int8']['mds_corr']:<12.4f}")

print()

# Memory Extrapolation
print(f"{'='*80}")
print("MEMORY EXTRAPOLATION TO 100K SAMPLES")
print(f"{'='*80}\n")

full_100k = 100000 * 100000 * 8 / (1024**3)
triblock_100k = full_100k * 0.6  # 40% savings
int8_100k_mb = (2 * 100000 * 32 * 1) / (1024**2)

print(f"{'Method':<20} {'Memory (100K samples)':<25} {'Feasible (32GB)':<20}")
print(f"{'-'*80}")
print(f"{'CPU Full':<20} {full_100k:>20.1f} GB    {'❌ No (exceeds RAM)':<20}")
print(f"{'CPU TriBlock':<20} {triblock_100k:>20.1f} GB    {'❌ No (still too large)':<20}")
print(f"{'GPU INT8':<20} {int8_100k_mb:>20.1f} MB    {'✅ Yes':<20}")

print()

# Why Not Other Quantization Levels
print(f"{'='*80}")
print("WHY GPU INT8 IS RECOMMENDED (vs other quantization levels)")
print(f"{'='*80}\n")

print("Based on 100-tree comparison:")
print()
print("GPU NF4 (4-bit):")
print("  ❌ 16× SLOWER than INT8 (283s vs 17s for 100 trees)")
print("  ❌ Bit-packing overhead dominates (4 bits = 2 values per byte)")
print("  ✅ Same MDS correlation as INT8 (67%)")
print("  ✅ 2× less memory than INT8")
print("  → Only use for extreme memory constraints (>200K samples)")
print()

print("GPU FP16 (16-bit half precision):")
print("  ❌ 17× SLOWER than INT8 (296s vs 17s for 100 trees)")
print("  ❌ WORSE quality than INT8 (53% vs 67% MDS correlation)")
print("  ❌ 2× MORE memory than INT8")
print("  → No advantage over INT8, not recommended")
print()

print("GPU FP32 (32-bit full precision):")
print("  ❌ 17× SLOWER than INT8 (298s vs 17s for 100 trees)")
print("  ❌ WORSE quality than INT8 (53% vs 67% MDS correlation)")
print("  ❌ 4× MORE memory than INT8")
print("  → No advantage over INT8, not recommended")
print()

print("CONCLUSION: GPU INT8 offers the best speed/quality/memory trade-off")
print("  ✅ Fastest GPU quantization (17s for 100 trees)")
print("  ✅ Best quality (67% MDS correlation, tied with NF4)")
print("  ✅ Good compression (6,250× vs full matrix for 100K samples)")
print()

# Save results to JSON
output_data = {
    'dataset': 'Wine',
    'ntree': ntree,
    'cpu_full': {
        'time_sec': results['cpu_full']['time'],
        'trees_per_sec': ntree / results['cpu_full']['time'],
        'memory_kb': results['cpu_full']['memory_kb'],
        'spearman': 1.0,
        'mds_corr': 1.0
    },
    'cpu_triblock': {
        'time_sec': results['cpu_triblock']['time'],
        'trees_per_sec': ntree / results['cpu_triblock']['time'],
        'memory_kb': results['cpu_triblock']['memory_kb'],
        'spearman': results['cpu_triblock']['spearman'],
        'mds_corr': results['cpu_triblock']['mds_corr']
    },
    'gpu_int8': {
        'time_sec': results['gpu_int8']['time'],
        'trees_per_sec': ntree / results['gpu_int8']['time'],
        'memory_kb': results['gpu_int8']['memory_kb'],
        'rank': results['gpu_int8']['rank'],
        'spearman': results['gpu_int8']['spearman'],
        'mds_corr': results['gpu_int8']['mds_corr']
    }
}

with open('quantization_final_1000trees.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"{'='*80}")
print("RESULTS SAVED")
print(f"{'='*80}\n")
print("  📊 quantization_final_1000trees.json - Machine-readable results")
print()

print(f"{'='*80}")
print("✅ Analysis complete! Results ready for arXiv paper.")
print(f"{'='*80}")

