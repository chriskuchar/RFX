#!/usr/bin/env python3
"""
Step 3: Wine Classification - Local Importance with Timing
Test configurations for 10 and 100 trees:
1. GPU Non-casewise
2. GPU Casewise
3. CPU Non-casewise
4. CPU Casewise
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.abspath('.'), 'python'))

import numpy as np
import RFX as rf
import urllib.request
import time
import json
from scipy.stats import spearmanr

def load_wine_cached(cache_file='wine.data'):
    """Load Wine dataset from UCI ML cache or download if needed"""
    
    # Check if cached
    if os.path.exists(cache_file):
        print(f"Loading cached Wine data from {cache_file}")
    else:
        print("Downloading Wine dataset from UCI...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
        urllib.request.urlretrieve(url, cache_file)
        print(f"Downloaded and cached to {cache_file}")
    
    # Load data (format: class,feature1,feature2,...,feature13)
    data = []
    with open(cache_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 14:
                data.append([float(x) for x in parts])
    
    data = np.array(data, dtype=np.float32)
    
    # Extract features and labels
    y = data[:, 0].astype(np.int32) - 1  # Classes are 1,2,3 -> convert to 0,1,2
    X = data[:, 1:]  # 13 features
    
    feature_names = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
                     'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
                     'Proanthocyanins', 'Color intensity', 'Hue',
                     'OD280/OD315 of diluted wines', 'Proline']
    
    return X, y, feature_names

def run_experiment(X, y, feature_names, ntree, configs):
    """Run experiment for given number of trees"""
    
    print("\n" + "=" * 80)
    print(f"RUNNING EXPERIMENT: {ntree} TREES")
    print("=" * 80)
    
    iseed = 42
    mtry = 4  # sqrt(13) ≈ 3.6
    batch_size_gpu = 0  # Auto-scaling
    
    results = {}
    timings = {}
    
    for config_name, config in configs.items():
        print("\n" + "-" * 80)
        print(f"Configuration: {config_name}")
        print("-" * 80)
        
        # Create model
        model = rf.RandomForestClassifier(
            ntree=ntree,
            mtry=mtry,
            iseed=iseed,
            use_gpu=config['use_gpu'],
            batch_size=batch_size_gpu if config['use_gpu'] else 1,
            use_casewise=config['use_casewise'],
            compute_importance=True,
            compute_local_importance=True
        )
        
        # Time training + importance
        start_time = time.time()
        model.fit(X, y)
        
        # Get importance (already computed during fit)
        overall_imp = model.feature_importances_()
        local_imp = model.get_local_importance()
        
        elapsed = time.time() - start_time
        
        print(f"Elapsed time: {elapsed:.2f}s ({ntree/elapsed:.2f} trees/sec)")
        
        print(f"\nOverall Importance (top 5):")
        top5_idx = np.argsort(overall_imp)[-5:][::-1]
        for idx in top5_idx:
            print(f"  {feature_names[idx]:35s}: {overall_imp[idx]:8.5f}")
        
        print(f"\nLocal Importance:")
        local_mean = np.mean(local_imp, axis=0)
        local_std = np.std(local_imp, axis=0)
        print(f"  Shape: {local_imp.shape}")
        print(f"  Mean per feature (top 5):")
        top5_mean_idx = np.argsort(local_mean)[-5:][::-1]
        for idx in top5_mean_idx:
            print(f"    {feature_names[idx]:35s}: {local_mean[idx]:8.5f} ± {local_std[idx]:8.5f}")
        
        results[config_name] = {
            'overall': overall_imp,
            'local': local_imp,
            'local_mean': local_mean,
            'local_std': local_std
        }
        
        timings[config_name] = {
            'elapsed_sec': elapsed,
            'trees_per_sec': ntree / elapsed
        }
    
    return results, timings

def print_comparison(results, ntree):
    """Print comparison summary"""
    
    print("\n" + "=" * 80)
    print(f"COMPARISON SUMMARY ({ntree} trees)")
    print("=" * 80)
    
    configs = list(results.keys())
    
    print("\n📊 Overall Importance Correlations:")
    for i, c1 in enumerate(configs):
        for c2 in configs[i+1:]:
            corr, pval = spearmanr(results[c1]['overall'], results[c2]['overall'])
            print(f"  {c1:20s} vs {c2:20s}: {corr:.4f} (p={pval:.4e})")
    
    print("\n📊 Local Importance Mean Correlations:")
    for i, c1 in enumerate(configs):
        for c2 in configs[i+1:]:
            corr, pval = spearmanr(results[c1]['local_mean'], results[c2]['local_mean'])
            print(f"  {c1:20s} vs {c2:20s}: {corr:.4f} (p={pval:.4e})")
    
    print("\n📊 Within-Platform Consistency:")
    # GPU: casewise vs non-casewise
    if 'GPU Non-casewise' in results and 'GPU Casewise' in results:
        corr_overall, _ = spearmanr(results['GPU Non-casewise']['overall'], results['GPU Casewise']['overall'])
        corr_local, _ = spearmanr(results['GPU Non-casewise']['local_mean'], results['GPU Casewise']['local_mean'])
        print(f"  GPU (CW vs NCW) Overall:     {corr_overall:.4f}")
        print(f"  GPU (CW vs NCW) Local Mean:  {corr_local:.4f}")
    
    # CPU: casewise vs non-casewise
    if 'CPU Non-casewise' in results and 'CPU Casewise' in results:
        corr_overall, _ = spearmanr(results['CPU Non-casewise']['overall'], results['CPU Casewise']['overall'])
        corr_local, _ = spearmanr(results['CPU Non-casewise']['local_mean'], results['CPU Casewise']['local_mean'])
        print(f"  CPU (CW vs NCW) Overall:     {corr_overall:.4f}")
        print(f"  CPU (CW vs NCW) Local Mean:  {corr_local:.4f}")
    
    print("\n📊 Cross-Platform Consistency:")
    # Non-casewise: GPU vs CPU
    if 'GPU Non-casewise' in results and 'CPU Non-casewise' in results:
        corr_overall, _ = spearmanr(results['GPU Non-casewise']['overall'], results['CPU Non-casewise']['overall'])
        corr_local, _ = spearmanr(results['GPU Non-casewise']['local_mean'], results['CPU Non-casewise']['local_mean'])
        print(f"  Non-casewise (GPU vs CPU) Overall:     {corr_overall:.4f}")
        print(f"  Non-casewise (GPU vs CPU) Local Mean:  {corr_local:.4f}")
    
    # Casewise: GPU vs CPU
    if 'GPU Casewise' in results and 'CPU Casewise' in results:
        corr_overall, _ = spearmanr(results['GPU Casewise']['overall'], results['CPU Casewise']['overall'])
        corr_local, _ = spearmanr(results['GPU Casewise']['local_mean'], results['CPU Casewise']['local_mean'])
        print(f"  Casewise (GPU vs CPU) Overall:         {corr_overall:.4f}")
        print(f"  Casewise (GPU vs CPU) Local Mean:      {corr_local:.4f}")

def main():
    print("=" * 80)
    print("STEP 3: LOCAL IMPORTANCE WITH TIMING - WINE DATASET")
    print("=" * 80)
    
    # Load Wine dataset
    X, y, feature_names = load_wine_cached()
    
    print(f"\nDataset: Wine (UCI ML)")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {len(np.unique(y))}")
    print(f"  Class distribution: {np.bincount(y)}")
    
    # Configurations
    configs = {
        'GPU Non-casewise': {'use_gpu': True, 'use_casewise': False},
        'GPU Casewise': {'use_gpu': True, 'use_casewise': True},
        'CPU Non-casewise': {'use_gpu': False, 'use_casewise': False},
        'CPU Casewise': {'use_gpu': False, 'use_casewise': True}
    }
    
    # Storage for all results
    all_results = {}
    all_timings = {}
    
    # ================================================================================
    # Experiment 1: 10 trees
    # ================================================================================
    ntree = 10
    results_10, timings_10 = run_experiment(X, y, feature_names, ntree, configs)
    all_results[ntree] = results_10
    all_timings[ntree] = timings_10
    print_comparison(results_10, ntree)
    
    # ================================================================================
    # Experiment 2: 100 trees
    # ================================================================================
    ntree = 100
    results_100, timings_100 = run_experiment(X, y, feature_names, ntree, configs)
    all_results[ntree] = results_100
    all_timings[ntree] = timings_100
    print_comparison(results_100, ntree)
    
    # ================================================================================
    # Final Summary
    # ================================================================================
    print("\n" + "=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)
    
    print("\n⏱️  10 Trees:")
    for config_name in configs.keys():
        t = timings_10[config_name]
        print(f"  {config_name:20s}: {t['elapsed_sec']:6.2f}s ({t['trees_per_sec']:6.2f} trees/sec)")
    
    print("\n⏱️  100 Trees:")
    for config_name in configs.keys():
        t = timings_100[config_name]
        print(f"  {config_name:20s}: {t['elapsed_sec']:6.2f}s ({t['trees_per_sec']:6.2f} trees/sec)")
    
    print("\n⚡ Speedup (GPU vs CPU):")
    for mode in ['Non-casewise', 'Casewise']:
        gpu_key = f'GPU {mode}'
        cpu_key = f'CPU {mode}'
        
        if gpu_key in timings_10 and cpu_key in timings_10:
            speedup_10 = timings_10[cpu_key]['elapsed_sec'] / timings_10[gpu_key]['elapsed_sec']
            print(f"  {mode:15s} (10 trees):  {speedup_10:.2f}x")
        
        if gpu_key in timings_100 and cpu_key in timings_100:
            speedup_100 = timings_100[cpu_key]['elapsed_sec'] / timings_100[gpu_key]['elapsed_sec']
            print(f"  {mode:15s} (100 trees): {speedup_100:.2f}x")
    
    # Save timings to JSON
    output = {
        'dataset': 'Wine',
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y)),
        'timings': {}
    }
    
    for ntree in [10, 100]:
        output['timings'][f'{ntree}_trees'] = {}
        for config_name in configs.keys():
            output['timings'][f'{ntree}_trees'][config_name] = all_timings[ntree][config_name]
    
    output_file = 'local_importance_timings.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Timings saved to {output_file}")
    
    print("\n" + "=" * 80)
    print("STEP 3 WINE DATASET TEST COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()

