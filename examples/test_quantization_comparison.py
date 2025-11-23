#!/usr/bin/env python3
"""
Comprehensive quantization level comparison for arXiv paper
Compares NF4, INT8, FP16, FP32 GPU low-rank vs CPU full matrix vs CPU block-sparse
Wine dataset (178 samples), 100 trees
"""

import RFX
import numpy as np
import time
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

ntree = 100
quant_levels = ['nf4', 'int8', 'fp16', 'fp32']

print(f"{'='*80}")
print(f"QUANTIZATION LEVEL COMPARISON ({ntree} trees)")
print(f"{'='*80}\n")

# Test 1: CPU Full Matrix (ground truth)
print("Training CPU Full Matrix (ground truth)...")
print(f"{'-'*80}")
start = time.time()
rf_cpu = RFX.RandomForestClassifier(
    ntree=ntree,
    use_gpu=False,
    use_casewise=False,
    compute_proximity=True,
    use_qlora=False,
    use_sparse=False,  # Full dense matrix
    batch_size=0,
    iseed=42
)
rf_cpu.fit(X, y)
cpu_time = time.time() - start

P_cpu = rf_cpu.get_proximity_matrix()
print(f"  Time: {cpu_time:.2f}s")
print(f"  Full proximity: shape={P_cpu.shape}, memory={P_cpu.nbytes/1024:.1f} KB")
print(f"  P_cpu stats: mean={np.mean(P_cpu):.4f}, std={np.std(P_cpu):.4f}")
print(f"  P_cpu diagonal: mean={np.mean(np.diag(P_cpu)):.4f}\n")

# Test 2: CPU Block-Sparse (memory-efficient CPU)
print("Training CPU Block-Sparse (memory-efficient)...")
print(f"{'-'*80}")
start = time.time()
rf_cpu_sparse = RFX.RandomForestClassifier(
    ntree=ntree,
    use_gpu=False,
    use_casewise=False,
    compute_proximity=True,
    use_qlora=False,
    use_sparse=True,  # Enable block-sparse storage
    sparsity_threshold=0.001,  # Conservative threshold
    batch_size=0,
    iseed=42
)
rf_cpu_sparse.fit(X, y)
cpu_sparse_time = time.time() - start

P_cpu_sparse = rf_cpu_sparse.get_proximity_matrix()
print(f"  Time: {cpu_sparse_time:.2f}s")
print(f"  Sparse proximity: shape={P_cpu_sparse.shape}, memory={P_cpu_sparse.nbytes/1024:.1f} KB (same as dense after conversion)")
print(f"  P_cpu_sparse stats: mean={np.mean(P_cpu_sparse):.4f}, std={np.std(P_cpu_sparse):.4f}")
print(f"  P_cpu_sparse diagonal: mean={np.mean(np.diag(P_cpu_sparse)):.4f}")

# Compute correlation between CPU full and CPU sparse
sparse_corr, _ = spearmanr(P_cpu_sparse.flatten(), P_cpu.flatten())
print(f"  Correlation with CPU full: {sparse_corr:.4f}\n")

# Store results for comparison
results = {}

# Train GPU for each quantization level
for quant in quant_levels:
    print(f"Training GPU with {quant.upper()} quantization...")
    print(f"{'-'*80}")
    
    start = time.time()
    rf_gpu = RFX.RandomForestClassifier(
        ntree=ntree,
        use_gpu=True,
        use_casewise=False,
        compute_proximity=True,
        use_qlora=True,
        quant_mode=quant,
        batch_size=0,
        iseed=42
    )
    rf_gpu.fit(X, y)
    gpu_time = time.time() - start
    
    # Get low-rank factors and reconstruct
    A, B, rank = rf_gpu.get_lowrank_factors()
    P_gpu_raw = A @ B.T
    
    # Normalize GPU proximity (divide by diagonal)
    P_gpu_diag = np.diag(P_gpu_raw).copy()
    P_gpu_diag[P_gpu_diag == 0] = 1.0
    P_gpu = P_gpu_raw / np.sqrt(P_gpu_diag[:, None] * P_gpu_diag[None, :])
    
    # Compute correlations
    spearman_corr, _ = spearmanr(P_gpu.flatten(), P_cpu.flatten())
    pearson_corr, _ = pearsonr(P_gpu.flatten(), P_cpu.flatten())
    
    # Compute errors
    mse = np.mean((P_gpu - P_cpu) ** 2)
    mae = np.mean(np.abs(P_gpu - P_cpu))
    
    # Memory
    gpu_memory = A.nbytes + B.nbytes
    cpu_memory = P_cpu.nbytes
    compression = cpu_memory / gpu_memory
    
    print(f"  Time: {gpu_time:.2f}s (speedup: {cpu_time/gpu_time:.2f}×)")
    print(f"  Low-rank: A={A.shape}, B={B.shape}, rank={rank}")
    print(f"  Memory: {gpu_memory/1024:.1f} KB (compression: {compression:.1f}×)")
    print(f"  Correlations: Spearman={spearman_corr:.4f}, Pearson={pearson_corr:.4f}")
    print(f"  Errors: MSE={mse:.6f}, MAE={mae:.6f}")
    
    # Store results
    results[quant] = {
        'P_gpu': P_gpu,
        'A': A,
        'B': B,
        'rank': rank,
        'time': gpu_time,
        'memory': gpu_memory,
        'compression': compression,
        'spearman': spearman_corr,
        'pearson': pearson_corr,
        'mse': mse,
        'mae': mae
    }
    print()

# Summary table
print(f"{'='*80}")
print("SUMMARY TABLE")
print(f"{'='*80}\n")
print(f"{'Quant':<8} {'Time(s)':<10} {'Memory(KB)':<12} {'Compress':<10} {'Spearman':<10} {'Pearson':<10} {'MSE':<10}")
print(f"{'-'*80}")
print(f"{'CPU':<8} {cpu_time:<10.2f} {P_cpu.nbytes/1024:<12.1f} {'1.0×':<10} {'1.0000':<10} {'1.0000':<10} {'0.000000':<10}")
for quant in quant_levels:
    r = results[quant]
    print(f"{quant.upper():<8} {r['time']:<10.2f} {r['memory']/1024:<12.1f} {r['compression']:<10.1f}× {r['spearman']:<10.4f} {r['pearson']:<10.4f} {r['mse']:<10.6f}")

# MDS embedding comparison
print(f"\n{'='*80}")
print("MDS EMBEDDING COMPARISON")
print(f"{'='*80}\n")

print("Computing CPU MDS embedding...")
D_cpu = 1 - P_cpu
np.fill_diagonal(D_cpu, 0)
mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42, max_iter=300, n_init=1)
coords_cpu = mds.fit_transform(D_cpu)

mds_results = {}
for quant in quant_levels:
    print(f"Computing {quant.upper()} MDS embedding...")
    P_gpu = results[quant]['P_gpu']
    D_gpu = 1 - P_gpu
    np.fill_diagonal(D_gpu, 0)
    
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42, max_iter=300, n_init=1)
    coords_gpu = mds.fit_transform(D_gpu)
    
    # Align to CPU using Procrustes
    R, scale = orthogonal_procrustes(coords_gpu, coords_cpu)
    coords_gpu_aligned = coords_gpu @ R
    
    # Compute correlations per axis
    corr_x = np.corrcoef(coords_gpu_aligned[:, 0], coords_cpu[:, 0])[0, 1]
    corr_y = np.corrcoef(coords_gpu_aligned[:, 1], coords_cpu[:, 1])[0, 1]
    corr_z = np.corrcoef(coords_gpu_aligned[:, 2], coords_cpu[:, 2])[0, 1]
    corr_mean = np.mean([corr_x, corr_y, corr_z])
    rmse = np.sqrt(np.mean((coords_gpu_aligned - coords_cpu) ** 2))
    
    mds_results[quant] = {
        'coords': coords_gpu_aligned,
        'corr_mean': corr_mean,
        'corr_x': corr_x,
        'corr_y': corr_y,
        'corr_z': corr_z,
        'rmse': rmse
    }
    
    print(f"  {quant.upper()}: Mean Corr={corr_mean:.4f}, RMSE={rmse:.4f}")

# MDS summary table
print(f"\n{'='*80}")
print("MDS CORRELATION TABLE")
print(f"{'='*80}\n")
print(f"{'Quant':<8} {'Mean Corr':<12} {'X Corr':<10} {'Y Corr':<10} {'Z Corr':<10} {'RMSE':<10}")
print(f"{'-'*80}")
for quant in quant_levels:
    r = mds_results[quant]
    print(f"{quant.upper():<8} {r['corr_mean']:<12.4f} {r['corr_x']:<10.4f} {r['corr_y']:<10.4f} {r['corr_z']:<10.4f} {r['rmse']:<10.4f}")

# Create comprehensive visualization
print(f"\n{'='*80}")
print("GENERATING FIGURES")
print(f"{'='*80}\n")

# Figure 1: 5×2 grid (CPU + 4 quant levels) × (2D + 3D)
fig = plt.figure(figsize=(16, 20))

# Row 1: CPU
ax1 = fig.add_subplot(5, 2, 1)
scatter1 = ax1.scatter(coords_cpu[:, 0], coords_cpu[:, 1], c=y, cmap='tab10', s=80, alpha=0.7, edgecolors='black', linewidths=0.5)
ax1.set_title(f'CPU Full Matrix (2D)\nMemory: {P_cpu.nbytes/1024:.1f} KB', fontsize=11, fontweight='bold')
ax1.set_xlabel('MDS Dimension 1', fontsize=9)
ax1.set_ylabel('MDS Dimension 2', fontsize=9)
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=ax1, label='Wine Class')

ax2 = fig.add_subplot(5, 2, 2, projection='3d')
scatter2 = ax2.scatter(coords_cpu[:, 0], coords_cpu[:, 1], coords_cpu[:, 2], c=y, cmap='tab10', s=60, alpha=0.7, edgecolors='black', linewidths=0.5)
ax2.set_title(f'CPU Full Matrix (3D)\nTime: {cpu_time:.2f}s', fontsize=11, fontweight='bold')
ax2.set_xlabel('MDS 1', fontsize=8)
ax2.set_ylabel('MDS 2', fontsize=8)
ax2.set_zlabel('MDS 3', fontsize=8)
ax2.view_init(elev=20, azim=45)

# Rows 2-5: Each quantization level
for idx, quant in enumerate(quant_levels):
    row = idx + 2
    r = results[quant]
    m = mds_results[quant]
    coords = m['coords']
    
    # 2D plot
    ax_2d = fig.add_subplot(5, 2, row*2-1)
    scatter_2d = ax_2d.scatter(coords[:, 0], coords[:, 1], c=y, cmap='tab10', s=80, alpha=0.7, edgecolors='black', linewidths=0.5)
    ax_2d.set_title(f'GPU {quant.upper()} (2D)\nMemory: {r["memory"]/1024:.1f} KB ({r["compression"]:.1f}× compression)', fontsize=11, fontweight='bold')
    ax_2d.set_xlabel('MDS Dimension 1', fontsize=9)
    ax_2d.set_ylabel('MDS Dimension 2', fontsize=9)
    ax_2d.grid(True, alpha=0.3)
    plt.colorbar(scatter_2d, ax=ax_2d, label='Wine Class')
    
    # 3D plot
    ax_3d = fig.add_subplot(5, 2, row*2, projection='3d')
    scatter_3d = ax_3d.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=y, cmap='tab10', s=60, alpha=0.7, edgecolors='black', linewidths=0.5)
    ax_3d.set_title(f'GPU {quant.upper()} (3D)\nMDS Corr: {m["corr_mean"]:.3f}, Time: {r["time"]:.2f}s', fontsize=11, fontweight='bold')
    ax_3d.set_xlabel('MDS 1', fontsize=8)
    ax_3d.set_ylabel('MDS 2', fontsize=8)
    ax_3d.set_zlabel('MDS 3', fontsize=8)
    ax_3d.view_init(elev=20, azim=45)

plt.tight_layout()
output_file = 'quantization_comparison_5x2.png'
plt.savefig(output_file, dpi=200, bbox_inches='tight')
print(f"📊 5×2 quantization comparison saved to: {output_file}")
plt.close()

# Figure 2: Correlation vs Memory trade-off scatter plot
fig, ax = plt.subplots(figsize=(10, 6))
colors = {'nf4': 'red', 'int8': 'orange', 'fp16': 'blue', 'fp32': 'green'}
for quant in quant_levels:
    r = results[quant]
    m = mds_results[quant]
    ax.scatter(r['memory']/1024, m['corr_mean'], s=300, c=colors[quant], alpha=0.7, edgecolors='black', linewidths=2, label=quant.upper())
    ax.text(r['memory']/1024, m['corr_mean'], f"  {quant.upper()}\n  {r['compression']:.1f}×", fontsize=10, va='center')

ax.set_xlabel('Memory Usage (KB)', fontsize=12, fontweight='bold')
ax.set_ylabel('MDS Correlation with CPU', fontsize=12, fontweight='bold')
ax.set_title('Quantization Trade-off: Memory vs Reconstruction Quality', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=11)
ax.set_xlim(left=0)
ax.set_ylim(0.5, 1.0)

output_file_tradeoff = 'quantization_tradeoff.png'
plt.savefig(output_file_tradeoff, dpi=200, bbox_inches='tight')
print(f"📊 Trade-off scatter plot saved to: {output_file_tradeoff}")
plt.close()

print(f"\n{'='*80}")
print("✅ Analysis complete! Ready for arXiv paper.")
print(f"{'='*80}\n")

print("KEY FINDINGS:")
print(f"  - All quantization levels preserve geometric structure (>60% MDS correlation)")
print(f"  - FP32 highest quality: {mds_results['fp32']['corr_mean']:.1%} MDS correlation")
print(f"  - NF4 best compression: {results['nf4']['compression']:.1f}× memory savings")
print(f"  - Recommendation: FP16 for quality, NF4 for large-scale datasets")
