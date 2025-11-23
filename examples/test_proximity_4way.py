"""
Test proximity computation for all 4 configurations:
- GPU Non-casewise (NF4 quantization, low-rank)
- GPU Casewise (NF4 quantization, low-rank)
- CPU Non-casewise (full matrix)
- CPU Casewise (full matrix)

Compare proximity matrices and verify correctness
"""

import RFX
import numpy as np
import urllib.request
import os
import time
from scipy.stats import spearmanr

# Load Wine dataset
cache_file = "wine_dataset.npz"
if os.path.exists(cache_file):
    data = np.load(cache_file)
    X, y = data['X'], data['y']
else:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    data = urllib.request.urlopen(url).read().decode('utf-8')
    lines = [line.strip() for line in data.split('\n') if line.strip()]
    data_list = [[float(x) for x in line.split(',')] for line in lines]
    data_array = np.array(data_list)
    y = data_array[:, 0].astype(int)
    X = data_array[:, 1:]
    np.savez(cache_file, X=X, y=y)

print(f"Wine dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes\n")

# Test configurations
ntree = 100
configs = [
    {"name": "GPU Non-casewise (NF4)", "use_gpu": True, "use_casewise": False, "quant_mode": "nf4"},
    {"name": "GPU Casewise (NF4)", "use_gpu": True, "use_casewise": True, "quant_mode": "nf4"},
    {"name": "CPU Non-casewise", "use_gpu": False, "use_casewise": False, "quant_mode": "fp32"},
    {"name": "CPU Casewise", "use_gpu": False, "use_casewise": True, "quant_mode": "fp32"},
]

results = []

print(f"{'='*80}")
print(f"TESTING PROXIMITY FOR ALL 4 CONFIGURATIONS ({ntree} trees)")
print(f"{'='*80}\n")

for config in configs:
    print(f"\n{'-'*80}")
    print(f"Testing: {config['name']}")
    print(f"{'-'*80}")
    
    try:
        # Train model with proximity
        rf = RFX.RandomForestClassifier(
            ntree=ntree,
            use_gpu=config['use_gpu'],
            use_casewise=config['use_casewise'],
            compute_proximity=True,
            quant_mode=config['quant_mode'],
            use_qlora=config['use_gpu'],  # GPU uses low-rank (QLORA)
            batch_size=0,  # Auto-scaling
            iseed=42  # Fixed seed for reproducibility
        )
        
        start_time = time.time()
        rf.fit(X, y)
        elapsed = time.time() - start_time
        
        # Get proximity matrix
        if config['use_gpu']:
            # GPU uses low-rank factors (A, B, rank)
            A, B, rank = rf.get_lowrank_factors()
            print(f"  Low-rank factors: A={A.shape}, B={B.shape}, rank={rank}")
            # Reconstruct proximity matrix from low-rank factors: P = A @ B.T
            prox = A @ B.T
        else:
            # CPU returns full matrix
            prox = rf.get_proximity_matrix()
        
        # Compute statistics
        prox_mean = np.mean(prox)
        prox_std = np.std(prox)
        prox_min = np.min(prox)
        prox_max = np.max(prox)
        prox_diag_mean = np.mean(np.diag(prox))
        
        # Check if proximity is symmetric
        is_symmetric = np.allclose(prox, prox.T, atol=1e-4)
        
        # Check for NaN or Inf
        has_nan = np.any(np.isnan(prox))
        has_inf = np.any(np.isinf(prox))
        
        print(f"\n✅ SUCCESS!")
        print(f"  Training + proximity time: {elapsed:.2f}s")
        print(f"  Trees/sec: {ntree/elapsed:.2f}")
        print(f"  Proximity shape: {prox.shape}")
        print(f"  Mean: {prox_mean:.6f}")
        print(f"  Std:  {prox_std:.6f}")
        print(f"  Range: [{prox_min:.6f}, {prox_max:.6f}]")
        print(f"  Diagonal mean: {prox_diag_mean:.6f} (should be ~1.0)")
        print(f"  Symmetric: {is_symmetric}")
        print(f"  Has NaN: {has_nan}")
        print(f"  Has Inf: {has_inf}")
        
        if config['use_gpu']:
            print(f"  Low-rank: True (NF4 quantization)")
        else:
            print(f"  Full matrix: True (FP32)")
        
        results.append({
            'name': config['name'],
            'config': config,
            'success': True,
            'elapsed': elapsed,
            'trees_per_sec': ntree/elapsed,
            'prox_mean': prox_mean,
            'prox_std': prox_std,
            'prox_min': prox_min,
            'prox_max': prox_max,
            'prox_diag_mean': prox_diag_mean,
            'is_symmetric': is_symmetric,
            'has_nan': has_nan,
            'has_inf': has_inf,
            'prox_matrix': prox
        })
        
    except Exception as e:
        print(f"\n❌ FAILED!")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        results.append({
            'name': config['name'],
            'config': config,
            'success': False,
            'error': str(e)
        })

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}\n")

successful = [r for r in results if r['success']]
failed = [r for r in results if not r['success']]

print(f"✅ Successful: {len(successful)}/{len(results)}")
print(f"❌ Failed: {len(failed)}/{len(results)}\n")

if successful:
    print(f"{'Configuration':<30} {'Time (s)':<12} {'Trees/sec':<12} {'Mean':<12} {'Diag':<12}")
    print(f"{'-'*80}")
    for r in successful:
        print(f"{r['name']:<30} {r['elapsed']:<12.2f} {r['trees_per_sec']:<12.2f} {r['prox_mean']:<12.6f} {r['prox_diag_mean']:<12.6f}")

# Compare proximity matrices
if len(successful) >= 2:
    print(f"\n{'='*80}")
    print("PROXIMITY MATRIX COMPARISON")
    print(f"{'='*80}\n")
    
    # Within-platform comparisons
    gpu_ncw = next((r for r in successful if 'GPU Non-casewise' in r['name']), None)
    gpu_cw = next((r for r in successful if 'GPU Casewise' in r['name']), None)
    cpu_ncw = next((r for r in successful if 'CPU Non-casewise' in r['name']), None)
    cpu_cw = next((r for r in successful if 'CPU Casewise' in r['name']), None)
    
    comparisons = []
    
    if gpu_ncw and gpu_cw:
        prox1_flat = gpu_ncw['prox_matrix'].flatten()
        prox2_flat = gpu_cw['prox_matrix'].flatten()
        corr, _ = spearmanr(prox1_flat, prox2_flat)
        mse = np.mean((prox1_flat - prox2_flat) ** 2)
        comparisons.append(("GPU Non-casewise vs GPU Casewise", corr, mse))
    
    if cpu_ncw and cpu_cw:
        prox1_flat = cpu_ncw['prox_matrix'].flatten()
        prox2_flat = cpu_cw['prox_matrix'].flatten()
        corr, _ = spearmanr(prox1_flat, prox2_flat)
        mse = np.mean((prox1_flat - prox2_flat) ** 2)
        comparisons.append(("CPU Non-casewise vs CPU Casewise", corr, mse))
    
    if gpu_ncw and cpu_ncw:
        prox1_flat = gpu_ncw['prox_matrix'].flatten()
        prox2_flat = cpu_ncw['prox_matrix'].flatten()
        corr, _ = spearmanr(prox1_flat, prox2_flat)
        mse = np.mean((prox1_flat - prox2_flat) ** 2)
        comparisons.append(("GPU Non-casewise vs CPU Non-casewise", corr, mse))
    
    if gpu_cw and cpu_cw:
        prox1_flat = gpu_cw['prox_matrix'].flatten()
        prox2_flat = cpu_cw['prox_matrix'].flatten()
        corr, _ = spearmanr(prox1_flat, prox2_flat)
        mse = np.mean((prox1_flat - prox2_flat) ** 2)
        comparisons.append(("GPU Casewise vs CPU Casewise", corr, mse))
    
    print(f"{'Comparison':<50} {'Correlation':<15} {'MSE':<15}")
    print(f"{'-'*80}")
    for comp_name, corr, mse in comparisons:
        print(f"{comp_name:<50} {corr:<15.4f} {mse:<15.9f}")
    
    print(f"\nInterpretation:")
    print(f"  - Correlation > 0.95: Excellent agreement")
    print(f"  - Correlation 0.80-0.95: Good agreement")
    print(f"  - Correlation < 0.80: Significant differences (expected for GPU vs CPU)")
    print(f"  - MSE < 0.001: Very similar")
    print(f"  - MSE > 0.01: Notable differences")

# Performance comparison
if len(successful) >= 2:
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*80}\n")
    
    if gpu_ncw and cpu_ncw:
        speedup = cpu_ncw['trees_per_sec'] / gpu_ncw['trees_per_sec']
        print(f"CPU Non-casewise vs GPU Non-casewise:")
        print(f"  CPU: {cpu_ncw['trees_per_sec']:.2f} trees/sec")
        print(f"  GPU: {gpu_ncw['trees_per_sec']:.2f} trees/sec")
        print(f"  Speedup: {speedup:.2f}x {'(CPU faster)' if speedup > 1 else '(GPU faster)'}\n")
    
    if gpu_cw and cpu_cw:
        speedup = cpu_cw['trees_per_sec'] / gpu_cw['trees_per_sec']
        print(f"CPU Casewise vs GPU Casewise:")
        print(f"  CPU: {cpu_cw['trees_per_sec']:.2f} trees/sec")
        print(f"  GPU: {gpu_cw['trees_per_sec']:.2f} trees/sec")
        print(f"  Speedup: {speedup:.2f}x {'(CPU faster)' if speedup > 1 else '(GPU faster)'}")

if failed:
    print(f"\n{'='*80}")
    print("FAILED CONFIGURATIONS")
    print(f"{'='*80}\n")
    for r in failed:
        print(f"  {r['name']}: {r['error']}")

print(f"\n{'='*80}")
print(f"All configurations {'✅ WORKING' if len(failed) == 0 else '⚠️ NEED ATTENTION'}")
print(f"{'='*80}")

