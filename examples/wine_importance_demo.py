#!/usr/bin/env python3
"""
Wine Classification - Overall and Local Variable Importance
Tests all 4 configurations: GPU/CPU × Casewise/Non-casewise
Shows overall importance rankings and local importance statistics
"""

import numpy as np
import rfx as rf
import time
from scipy.stats import spearmanr

# Feature names for Wine dataset
FEATURE_NAMES = [
    'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
    'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
    'Proanthocyanins', 'Color intensity', 'Hue',
    'OD280/OD315 of diluted wines', 'Proline'
]

CLASS_NAMES = ['Class 0', 'Class 1', 'Class 2']

def run_experiment(X, y, n_classes, use_gpu, use_casewise, ntree=100):
    """Run a single experiment and return results"""
    
    mode = "GPU" if use_gpu else "CPU"
    weighting = "Casewise" if use_casewise else "Non-casewise"
    
    print(f"\n{'='*70}")
    print(f"  {mode} {weighting}")
    print(f"{'='*70}")
    
    # Create model with importance enabled
    model = rf.RandomForestClassifier(
        ntree=ntree,
        mtry=4,
        nsample=X.shape[0],
        nclass=n_classes,
        use_gpu=use_gpu,
        batch_size=0,  # Auto SM-aware batching
        iseed=42,
        compute_proximity=False,
        compute_importance=True,        # Overall importance
        compute_local_importance=True,  # Local importance (per-sample)
        use_casewise=use_casewise
    )
    
    # Train
    print(f"\nTraining {ntree} trees with importance computation...")
    start_time = time.time()
    model.fit(X, y)
    elapsed = time.time() - start_time
    
    # Get results
    oob_error = model.get_oob_error()
    oob_preds = model.get_oob_predictions()
    overall_imp = model.feature_importances_()
    local_imp = model.get_local_importance()
    
    print(f"Training time: {elapsed:.2f}s ({ntree/elapsed:.1f} trees/sec)")
    print(f"OOB Error: {oob_error:.4f} ({oob_error*100:.2f}%)")
    
    # Confusion Matrix (built-in)
    cm = rf.confusion_matrix(y.astype(np.int32), oob_preds.astype(np.int32))
    print(f"\nConfusion Matrix (rf.confusion_matrix):")
    print(cm)
    
    # Classification Report (built-in)
    print(f"\nClassification Report (rf.classification_report):")
    print(rf.classification_report(y.astype(np.int32), oob_preds.astype(np.int32)))
    
    # Overall Importance
    print(f"\nOverall Feature Importance (Top 10):")
    sorted_idx = np.argsort(overall_imp)[::-1]
    print(f"   {'Rank':<5} {'Feature':<35} {'Importance':>12}")
    print(f"   {'-'*55}")
    for rank, idx in enumerate(sorted_idx[:10], 1):
        print(f"   {rank:<5} {FEATURE_NAMES[idx]:<35} {overall_imp[idx]:>12.6f}")
    
    # Local Importance Statistics
    print(f"\nLocal Importance (per-sample):")
    print(f"   Shape: {local_imp.shape} (samples × features)")
    
    local_mean = np.mean(local_imp, axis=0)
    local_std = np.std(local_imp, axis=0)
    local_min = np.min(local_imp, axis=0)
    local_max = np.max(local_imp, axis=0)
    
    print(f"\n   Feature-wise Local Importance Statistics (Top 5 by mean):")
    sorted_local_idx = np.argsort(local_mean)[::-1]
    print(f"   {'Feature':<35} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"   {'-'*75}")
    for idx in sorted_local_idx[:5]:
        print(f"   {FEATURE_NAMES[idx]:<35} {local_mean[idx]:>10.6f} {local_std[idx]:>10.6f} "
              f"{local_min[idx]:>10.6f} {local_max[idx]:>10.6f}")
    
    # Sample-wise statistics
    sample_imp_mean = np.mean(local_imp, axis=1)
    print(f"\n   Sample-wise Local Importance:")
    print(f"   - Mean across all samples: {sample_imp_mean.mean():.6f}")
    print(f"   - Std across all samples:  {sample_imp_mean.std():.6f}")
    print(f"   - Min sample importance:   {sample_imp_mean.min():.6f}")
    print(f"   - Max sample importance:   {sample_imp_mean.max():.6f}")
    
    # Top 5 most "important" samples
    top_samples = np.argsort(sample_imp_mean)[::-1][:5]
    print(f"\n   Top 5 Most Important Samples:")
    for rank, idx in enumerate(top_samples, 1):
        print(f"   {rank}. Sample {idx} (class {y[idx]}): {sample_imp_mean[idx]:.6f}")
    
    return {
        'mode': f"{mode} {weighting}",
        'oob_error': oob_error,
        'overall_imp': overall_imp,
        'local_imp': local_imp,
        'local_mean': local_mean,
        'time': elapsed
    }

def main():
    print("=" * 70)
    print("  WINE CLASSIFICATION - OVERALL AND LOCAL IMPORTANCE")
    print("  Testing: GPU/CPU × Casewise/Non-casewise")
    print("=" * 70)
    
    # Load Wine dataset (built-in)
    X, y = rf.load_wine()
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    
    print(f"\nDataset: Wine (UCI ML - built-in)")
    print(f"   Samples: {n_samples}")
    print(f"   Features: {n_features}")
    print(f"   Classes: {n_classes}")
    
    # Run all 4 configurations
    ntree = 100
    results = {}
    
    # 1. GPU Non-casewise
    results['gpu_ncw'] = run_experiment(X, y, n_classes, use_gpu=True, use_casewise=False, ntree=ntree)
    
    # 2. GPU Casewise
    results['gpu_cw'] = run_experiment(X, y, n_classes, use_gpu=True, use_casewise=True, ntree=ntree)
    
    # 3. CPU Non-casewise
    results['cpu_ncw'] = run_experiment(X, y, n_classes, use_gpu=False, use_casewise=False, ntree=ntree)
    
    # 4. CPU Casewise
    results['cpu_cw'] = run_experiment(X, y, n_classes, use_gpu=False, use_casewise=True, ntree=ntree)
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("  IMPORTANCE COMPARISON")
    print("=" * 70)
    
    # Overall importance correlation matrix
    print("\nOverall Importance Spearman Correlations:")
    configs = list(results.keys())
    print(f"   {'':25s}", end="")
    for c in configs:
        print(f"{results[c]['mode'][:12]:>14s}", end="")
    print()
    
    for c1 in configs:
        print(f"   {results[c1]['mode']:<25s}", end="")
        for c2 in configs:
            corr, _ = spearmanr(results[c1]['overall_imp'], results[c2]['overall_imp'])
            print(f"{corr:>14.4f}", end="")
        print()
    
    # Local importance mean correlation matrix
    print("\nLocal Importance (Mean) Spearman Correlations:")
    print(f"   {'':25s}", end="")
    for c in configs:
        print(f"{results[c]['mode'][:12]:>14s}", end="")
    print()
    
    for c1 in configs:
        print(f"   {results[c1]['mode']:<25s}", end="")
        for c2 in configs:
            corr, _ = spearmanr(results[c1]['local_mean'], results[c2]['local_mean'])
            print(f"{corr:>14.4f}", end="")
        print()
    
    # Top features comparison
    print("\nTop 3 Features by Overall Importance:")
    print(f"   {'Configuration':<25s} {'#1':<20s} {'#2':<20s} {'#3':<20s}")
    print(f"   {'-'*85}")
    for key, res in results.items():
        sorted_idx = np.argsort(res['overall_imp'])[::-1]
        top3 = [FEATURE_NAMES[i][:18] for i in sorted_idx[:3]]
        print(f"   {res['mode']:<25s} {top3[0]:<20s} {top3[1]:<20s} {top3[2]:<20s}")
    
    print("\n" + "=" * 70)
    print("  TEST COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    main()

