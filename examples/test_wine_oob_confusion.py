#!/usr/bin/env python3
"""
Wine Classification - OOB Error, Confusion Matrix, Classification Report
Tests all 4 configurations: GPU/CPU × Casewise/Non-casewise
Uses built-in load_wine, confusion_matrix, and classification_report
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))

import numpy as np
import RFX as rf
import time

# Feature names for Wine dataset
FEATURE_NAMES = [
    'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
    'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
    'Proanthocyanins', 'Color intensity', 'Hue',
    'OD280/OD315 of diluted wines', 'Proline'
]

CLASS_NAMES = ['Class 0', 'Class 1', 'Class 2']

def print_confusion_matrix(cm, n_classes):
    """Pretty print confusion matrix"""
    # Header
    header = "          " + "  ".join(f"Pred {i}" for i in range(n_classes))
    print(header)
    print("-" * len(header))
    
    # Rows
    for i in range(n_classes):
        row = f"True {i}  |"
        for j in range(n_classes):
            row += f"   {cm[i, j]:3d}  "
        print(row)
    print()

def run_experiment(X, y, n_classes, use_gpu, use_casewise, ntree=100):
    """Run a single experiment and return results"""
    
    mode = "GPU" if use_gpu else "CPU"
    weighting = "Casewise" if use_casewise else "Non-casewise"
    
    print(f"\n{'='*70}")
    print(f"  {mode} {weighting}")
    print(f"{'='*70}")
    
    # Create model
    model = rf.RandomForestClassifier(
        ntree=ntree,
        mtry=4,  # sqrt(13) ≈ 3.6
        nsample=X.shape[0],
        nclass=n_classes,
        use_gpu=use_gpu,
        batch_size=0,  # Auto SM-aware batching
        iseed=42,
        compute_proximity=False,
        compute_importance=True,
        compute_local_importance=False,
        use_casewise=use_casewise
    )
    
    # Train
    print(f"\nTraining {ntree} trees...")
    start_time = time.time()
    model.fit(X, y)
    elapsed = time.time() - start_time
    
    # Get results
    oob_error = model.get_oob_error()
    oob_preds = model.get_oob_predictions()
    
    print(f"Training time: {elapsed:.2f}s ({ntree/elapsed:.1f} trees/sec)")
    
    # OOB Error
    print(f"\n📊 OOB Error: {oob_error:.6f} ({oob_error*100:.2f}%)")
    print(f"   OOB Accuracy: {(1-oob_error)*100:.2f}%")
    
    # Confusion Matrix (using RFX built-in)
    cm = rf.confusion_matrix(y.astype(np.int32), oob_preds.astype(np.int32))
    print(f"\n📊 Confusion Matrix (rf.confusion_matrix):")
    print_confusion_matrix(cm, n_classes)
    
    # Classification Report (using RFX built-in)
    report_str = rf.classification_report(y.astype(np.int32), oob_preds.astype(np.int32))
    print(f"📊 Classification Report (rf.classification_report):")
    print(report_str)
    
    return {
        'mode': f"{mode} {weighting}",
        'oob_error': oob_error,
        'confusion_matrix': cm,
        'time': elapsed,
        'oob_preds': oob_preds
    }

def main():
    print("=" * 70)
    print("  WINE CLASSIFICATION - OOB ERROR, CONFUSION MATRIX, CLASSIFICATION REPORT")
    print("  Testing: GPU/CPU × Casewise/Non-casewise")
    print("=" * 70)
    
    # Load Wine dataset (built-in)
    X, y = rf.load_wine()
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    
    print(f"\n📂 Dataset: Wine (UCI ML - built-in)")
    print(f"   Samples: {n_samples}")
    print(f"   Features: {n_features}")
    print(f"   Classes: {n_classes}")
    print(f"   Class distribution: {np.bincount(y).tolist()}")
    
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
    print("  SUMMARY COMPARISON")
    print("=" * 70)
    
    print("\n📊 OOB Errors:")
    print(f"   {'Configuration':<25s} {'OOB Error':>12s} {'Accuracy':>12s} {'Time':>10s}")
    print("   " + "-" * 60)
    for key, res in results.items():
        print(f"   {res['mode']:<25s} {res['oob_error']:>12.6f} {(1-res['oob_error'])*100:>11.2f}% {res['time']:>9.2f}s")
    
    # Note: Detailed F1 scores are printed in the classification report above
    print("\n📊 Quick Summary:")
    
    print("\n📊 Prediction Agreement (vs GPU Non-casewise):")
    baseline = results['gpu_ncw']['oob_preds']
    for key, res in results.items():
        if key != 'gpu_ncw':
            agreement = np.sum(res['oob_preds'] == baseline) / len(baseline)
            print(f"   {res['mode']:<25s} {agreement*100:>6.2f}% agreement")
    
    print("\n📊 Casewise vs Non-casewise Differences:")
    gpu_diff = abs(results['gpu_cw']['oob_error'] - results['gpu_ncw']['oob_error'])
    cpu_diff = abs(results['cpu_cw']['oob_error'] - results['cpu_ncw']['oob_error'])
    print(f"   GPU:  {gpu_diff:.6f} ({gpu_diff*100:.2f}% difference)")
    print(f"   CPU:  {cpu_diff:.6f} ({cpu_diff*100:.2f}% difference)")
    
    if gpu_diff < 0.001 and cpu_diff < 0.001:
        print("\n   ⚠️  WARNING: Casewise and non-casewise produce IDENTICAL results!")
    else:
        print("\n   ✅ Casewise and non-casewise produce DIFFERENT results (expected!)")
    
    print("\n" + "=" * 70)
    print("  TEST COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    main()

