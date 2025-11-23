#!/usr/bin/env python3
"""
Generate OOB convergence vs. test set error curve for Wine dataset up to 10,000 trees.
Uses GPU for fast training with SM-aware auto-scaling.
"""

import numpy as np
import sys
import os
import time

# Add build directory to path if running from examples folder
# For installed package, this is not needed
build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build')
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import RFX
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def load_wine_data():
    """Load Wine dataset from UCI ML repository."""
    import urllib.request
    import os
    
    cache_file = 'wine.data'
    if not os.path.exists(cache_file):
        print("Downloading Wine dataset...")
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
        urllib.request.urlretrieve(url, cache_file)
    
    data = np.loadtxt(cache_file, delimiter=',')
    y = data[:, 0].astype(np.int32) - 1  # Classes 1,2,3 -> 0,1,2
    X = data[:, 1:].astype(np.float32)
    
    # Standardize features
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    return X, y

def train_and_track_oob(X_train, y_train, X_test, y_test, max_trees=1000, checkpoints=None):
    """
    Train multiple RF models at different tree counts to track OOB convergence.
    """
    if checkpoints is None:
        # Generate checkpoints: 10-tree intervals
        checkpoints = list(range(10, max_trees+1, 10))
    
    oob_errors = []
    test_errors = []
    tree_counts = []
    
    print(f"\nTraining models at various tree counts up to {max_trees}...")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}, Classes: {len(np.unique(y_train))}")
    print(f"Total checkpoints: {len([c for c in checkpoints if c <= max_trees])}")
    
    for i, ntrees in enumerate(checkpoints):
        if ntrees > max_trees:
            break
        
        start_time = time.time()
        
        # Train RF with ntrees
        rf = RFX.RandomForestClassifier(
            ntree=ntrees,
            mtry=int(np.sqrt(X_train.shape[1])),
            use_gpu=True,
            batch_size=0,  # Auto-scaling
            compute_importance=False,
            compute_proximity=False,
            use_casewise=False
        )
        rf.fit(X_train, y_train)
        
        elapsed = time.time() - start_time
        
        # Get OOB error
        oob_error = rf.oob_error()
        
        # Get test error
        y_pred = rf.predict(X_test)
        test_error = 1.0 - np.mean(y_pred == y_test)
        
        oob_errors.append(oob_error)
        test_errors.append(test_error)
        tree_counts.append(ntrees)
        
        trees_per_sec = ntrees / elapsed if elapsed > 0 else 0
        print(f"[{i+1:3d}] Trees: {ntrees:5d} | OOB: {oob_error:.4f} | "
              f"Test: {test_error:.4f} | Time: {elapsed:6.2f}s | "
              f"Speed: {trees_per_sec:6.1f} trees/s")
    
    return tree_counts, oob_errors, test_errors

def plot_convergence(tree_counts, oob_errors, test_errors, output_file='oob_convergence_1k.pdf'):
    """
    Generate publication-quality convergence plot.
    """
    # Calculate correlation
    correlation, p_value = pearsonr(oob_errors, test_errors)
    
    plt.figure(figsize=(10, 6))
    
    # Plot errors
    plt.plot(tree_counts, oob_errors, 'o-', color='#1f77b4', linewidth=2, 
             markersize=3, label='OOB Error', alpha=0.8)
    plt.plot(tree_counts, test_errors, 's-', color='#ff7f0e', linewidth=2, 
             markersize=3, label='Test Set Error', alpha=0.8)
    
    plt.xlabel('Number of Trees', fontsize=14, fontweight='bold')
    plt.ylabel('Classification Error', fontsize=14, fontweight='bold')
    plt.title('OOB Error Convergence vs. Test Set Error (Wine Dataset, 1000 Trees)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add correlation annotation
    plt.text(0.05, 0.95, f'Correlation: ρ = {correlation:.3f}', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))
    
    plt.xlim(0, max(tree_counts) * 1.05)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    plt.close()
    
    return correlation

def main():
    print("=" * 80)
    print("OOB Convergence Analysis: Wine Dataset (1,000 Trees)")
    print("=" * 80)
    
    # Load data
    X, y = load_wine_data()
    
    # Split into train/test (80/20)
    np.random.seed(42)
    n_samples = len(y)
    indices = np.random.permutation(n_samples)
    n_train = int(0.8 * n_samples)
    
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"\nDataset split:")
    print(f"  Training: {len(y_train)} samples")
    print(f"  Test: {len(y_test)} samples")
    
    # Train and track OOB
    overall_start = time.time()
    tree_counts, oob_errors, test_errors = train_and_track_oob(
        X_train, y_train, X_test, y_test, max_trees=1000
    )
    overall_elapsed = time.time() - overall_start
    
    print(f"\n{'=' * 80}")
    print(f"Total training time: {overall_elapsed:.2f}s")
    print(f"Average time per tree: {overall_elapsed / tree_counts[-1]:.4f}s")
    print(f"Trees per second: {tree_counts[-1] / overall_elapsed:.2f}")
    
    # Plot convergence
    correlation = plot_convergence(tree_counts, oob_errors, test_errors)
    
    # Save results to JSON
    import json
    results = {
        'dataset': 'Wine',
        'max_trees': tree_counts[-1],
        'train_samples': len(y_train),
        'test_samples': len(y_test),
        'features': X_train.shape[1],
        'classes': len(np.unique(y_train)),
        'total_time_sec': overall_elapsed,
        'trees_per_sec': tree_counts[-1] / overall_elapsed,
        'correlation': float(correlation),
        'final_oob_error': float(oob_errors[-1]),
        'final_test_error': float(test_errors[-1]),
        'tree_counts': tree_counts,
        'oob_errors': [float(e) for e in oob_errors],
        'test_errors': [float(e) for e in test_errors]
    }
    
    with open('oob_convergence_1k_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: oob_convergence_1k_results.json")
    print(f"\nFinal Results:")
    print(f"  OOB Error: {oob_errors[-1]:.4f}")
    print(f"  Test Error: {test_errors[-1]:.4f}")
    print(f"  Correlation: ρ = {correlation:.4f}")
    print("=" * 80)

if __name__ == '__main__':
    main()

