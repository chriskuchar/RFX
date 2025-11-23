#!/usr/bin/env python3
"""
Step 1: Wine Classification - OOB Error and Confusion Matrix
Using Wine dataset from UCI ML to test casewise vs non-casewise differences
Test configurations:
1. GPU Non-casewise (batch_size=1, 20 trees)
2. GPU Casewise (batch_size=1, 20 trees)
3. CPU Non-casewise (20 trees)
4. CPU Casewise (20 trees)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.abspath('.'), 'python'))

import numpy as np
import RFX as rf
import urllib.request
import signal
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

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

def compute_confusion_matrix(y_true, y_pred, n_classes):
    """Compute confusion matrix without sklearn"""
    conf_mat = np.zeros((n_classes, n_classes), dtype=np.int32)
    for true, pred in zip(y_true, y_pred):
        true_int = int(true)
        pred_int = int(pred)
        if 0 <= true_int < n_classes and 0 <= pred_int < n_classes:
            conf_mat[true_int, pred_int] += 1
    return conf_mat

# Common parameters (define BEFORE using them)
ntree = 100
iseed = 42
mtry = 4  # sqrt(13) ≈ 3.6
batch_size_gpu = 5  # Parallel mode - process 5 trees at once

print("=" * 80)
print(f"STEP 1: OOB ERROR AND CONFUSION MATRIX - WINE DATASET ({ntree} TREES)")
print("=" * 80)

# Load Wine dataset
X, y, feature_names = load_wine_cached()

print(f"\nDataset: Wine (UCI ML)")
print(f"  Samples: {X.shape[0]}")
print(f"  Features: {X.shape[1]}")
print(f"  Classes: {len(np.unique(y))}")
print(f"  Class distribution: {np.bincount(y)}")

n_classes = len(np.unique(y))

results = {}

# ============================================================================
# 1. GPU NON-CASEWISE
# ============================================================================
print("\n" + "=" * 80)
print(f"1. GPU NON-CASEWISE (batch_size={batch_size_gpu}, {ntree} trees)")
print("=" * 80)

print(f"[PYTHON DEBUG] Creating RandomForestClassifier with batch_size={batch_size_gpu}")

model_gpu_ncw = rf.RandomForestClassifier(
    ntree=ntree,
    mtry=mtry,
    nsample=X.shape[0],
    nclass=n_classes,
    use_gpu=True,
    batch_size=batch_size_gpu,
    iseed=iseed,
    compute_proximity=False,
    compute_importance=False,
    compute_local_importance=False,
    use_casewise=False
)

print("Training...")
try:
    with time_limit(10):
        model_gpu_ncw.fit(X, y)
    print("✓ Training completed")
except TimeoutException:
    print("✗ TIMEOUT: Training took longer than 10 seconds - HANGING!")
    print("   GPU parallel mode is hanging during tnodewt computation.")
    print("   This is a known issue with parallel mode + tnodewt on CPU.")
    sys.exit(1)

try:
    with time_limit(10):
        oob_err_gpu_ncw = model_gpu_ncw.get_oob_error()
    print("✓ OOB error computed")
except TimeoutException:
    print("✗ TIMEOUT: get_oob_error() took longer than 10 seconds - HANGING!")
    sys.exit(1)

try:
    with time_limit(10):
        oob_pred_gpu_ncw = model_gpu_ncw.get_oob_predictions()
        conf_mat_gpu_ncw = compute_confusion_matrix(y, oob_pred_gpu_ncw, n_classes)
    print("✓ OOB predictions and confusion matrix computed")
except TimeoutException:
    print("✗ TIMEOUT: get_oob_predictions() took longer than 10 seconds - HANGING!")
    sys.exit(1)

print(f"\nOOB Error: {oob_err_gpu_ncw:.6f} ({oob_err_gpu_ncw*100:.2f}%)")
print(f"OOB Accuracy: {(1-oob_err_gpu_ncw)*100:.2f}%")
print(f"\nConfusion Matrix:")
print(conf_mat_gpu_ncw)

results['gpu_ncw'] = {
    'oob_error': oob_err_gpu_ncw,
    'confusion_matrix': conf_mat_gpu_ncw,
    'oob_predictions': oob_pred_gpu_ncw
}

# ============================================================================
# 2. GPU CASEWISE
# ============================================================================
print("\n" + "=" * 80)
print(f"2. GPU CASEWISE (batch_size={batch_size_gpu}, {ntree} trees)")
print("=" * 80)

model_gpu_cw = rf.RandomForestClassifier(
    ntree=ntree,
    mtry=mtry,
    nsample=X.shape[0],
    nclass=n_classes,
    use_gpu=True,
    batch_size=batch_size_gpu,
    iseed=iseed,
    compute_proximity=False,
    compute_importance=False,
    compute_local_importance=False,
    use_casewise=True
)

print("Training...")
try:
    with time_limit(10):
        model_gpu_cw.fit(X, y)
    print("✓ Training completed")
except TimeoutException:
    print("✗ TIMEOUT: Training took longer than 10 seconds - HANGING!")
    print("   GPU parallel mode is hanging during tnodewt computation.")
    sys.exit(1)

try:
    with time_limit(10):
        oob_err_gpu_cw = model_gpu_cw.get_oob_error()
    print("✓ OOB error computed")
except TimeoutException:
    print("✗ TIMEOUT: get_oob_error() took longer than 10 seconds - HANGING!")
    sys.exit(1)

try:
    with time_limit(10):
        oob_pred_gpu_cw = model_gpu_cw.get_oob_predictions()
        conf_mat_gpu_cw = compute_confusion_matrix(y, oob_pred_gpu_cw, n_classes)
    print("✓ OOB predictions and confusion matrix computed")
except TimeoutException:
    print("✗ TIMEOUT: get_oob_predictions() took longer than 10 seconds - HANGING!")
    sys.exit(1)

print(f"\nOOB Error: {oob_err_gpu_cw:.6f} ({oob_err_gpu_cw*100:.2f}%)")
print(f"OOB Accuracy: {(1-oob_err_gpu_cw)*100:.2f}%")
print(f"\nConfusion Matrix:")
print(conf_mat_gpu_cw)

results['gpu_cw'] = {
    'oob_error': oob_err_gpu_cw,
    'confusion_matrix': conf_mat_gpu_cw,
    'oob_predictions': oob_pred_gpu_cw
}

# ============================================================================
# 3. CPU NON-CASEWISE
# ============================================================================
print("\n" + "=" * 80)
print(f"3. CPU NON-CASEWISE ({ntree} trees)")
print("=" * 80)

model_cpu_ncw = rf.RandomForestClassifier(
    ntree=ntree,
    mtry=mtry,
    nsample=X.shape[0],
    nclass=n_classes,
    use_gpu=False,
    iseed=iseed,
    compute_proximity=False,
    compute_importance=False,
    compute_local_importance=False,
    use_casewise=False
)

print("Training...")
model_cpu_ncw.fit(X, y)

oob_err_cpu_ncw = model_cpu_ncw.get_oob_error()
oob_pred_cpu_ncw = model_cpu_ncw.get_oob_predictions()
conf_mat_cpu_ncw = compute_confusion_matrix(y, oob_pred_cpu_ncw, n_classes)

print(f"\nOOB Error: {oob_err_cpu_ncw:.6f} ({oob_err_cpu_ncw*100:.2f}%)")
print(f"OOB Accuracy: {(1-oob_err_cpu_ncw)*100:.2f}%")
print(f"\nConfusion Matrix:")
print(conf_mat_cpu_ncw)

results['cpu_ncw'] = {
    'oob_error': oob_err_cpu_ncw,
    'confusion_matrix': conf_mat_cpu_ncw,
    'oob_predictions': oob_pred_cpu_ncw
}

# ============================================================================
# 4. CPU CASEWISE
# ============================================================================
print("\n" + "=" * 80)
print(f"4. CPU CASEWISE ({ntree} trees)")
print("=" * 80)

model_cpu_cw = rf.RandomForestClassifier(
    ntree=ntree,
    mtry=mtry,
    nsample=X.shape[0],
    nclass=n_classes,
    use_gpu=False,
    iseed=iseed,
    compute_proximity=False,
    compute_importance=False,
    compute_local_importance=False,
    use_casewise=True
)

print("Training...")
model_cpu_cw.fit(X, y)

oob_err_cpu_cw = model_cpu_cw.get_oob_error()
oob_pred_cpu_cw = model_cpu_cw.get_oob_predictions()
conf_mat_cpu_cw = compute_confusion_matrix(y, oob_pred_cpu_cw, n_classes)

print(f"\nOOB Error: {oob_err_cpu_cw:.6f} ({oob_err_cpu_cw*100:.2f}%)")
print(f"OOB Accuracy: {(1-oob_err_cpu_cw)*100:.2f}%")
print(f"\nConfusion Matrix:")
print(conf_mat_cpu_cw)

results['cpu_cw'] = {
    'oob_error': oob_err_cpu_cw,
    'confusion_matrix': conf_mat_cpu_cw,
    'oob_predictions': oob_pred_cpu_cw
}

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

print("\n📊 OOB Errors:")
print(f"  GPU Non-casewise: {results['gpu_ncw']['oob_error']:.6f} ({results['gpu_ncw']['oob_error']*100:.2f}%)")
print(f"  GPU Casewise:     {results['gpu_cw']['oob_error']:.6f} ({results['gpu_cw']['oob_error']*100:.2f}%)")
print(f"  CPU Non-casewise: {results['cpu_ncw']['oob_error']:.6f} ({results['cpu_ncw']['oob_error']*100:.2f}%)")
print(f"  CPU Casewise:     {results['cpu_cw']['oob_error']:.6f} ({results['cpu_cw']['oob_error']*100:.2f}%)")

print("\n🔍 CASEWISE vs NON-CASEWISE Differences:")
gpu_diff = abs(results['gpu_cw']['oob_error'] - results['gpu_ncw']['oob_error'])
cpu_diff = abs(results['cpu_cw']['oob_error'] - results['cpu_ncw']['oob_error'])

print(f"  GPU:  {gpu_diff:.6f} ({gpu_diff*100:.2f}% difference)")
print(f"  CPU:  {cpu_diff:.6f} ({cpu_diff*100:.2f}% difference)")

if gpu_diff < 0.001 and cpu_diff < 0.001:
    print("\n⚠️  WARNING: Casewise and non-casewise produce IDENTICAL results!")
    print("   This suggests casewise weighting is not being applied to OOB votes.")
elif gpu_diff < 0.001:
    print("\n⚠️  WARNING: GPU casewise = non-casewise (no difference)")
    print("   GPU casewise weighting may not be working correctly.")
elif cpu_diff < 0.001:
    print("\n⚠️  WARNING: CPU casewise = non-casewise (no difference)")
    print("   CPU casewise weighting may not be working correctly.")
else:
    print("\n✅ Casewise and non-casewise produce DIFFERENT results (expected!)")

print("\n📊 Confusion Matrix Differences (from CPU Non-casewise):")
baseline_conf = results['cpu_ncw']['confusion_matrix']
for name, data in results.items():
    if name != 'cpu_ncw':
        diff = np.abs(data['confusion_matrix'] - baseline_conf).sum()
        print(f"  {name}: {diff} total differences")

print("\n📊 Prediction Agreement (with CPU Non-casewise):")
baseline_pred = results['cpu_ncw']['oob_predictions']
for name, data in results.items():
    if name != 'cpu_ncw':
        agreement = np.sum(data['oob_predictions'] == baseline_pred) / len(baseline_pred)
        print(f"  {name}: {agreement*100:.2f}% agreement")

print("\n" + "=" * 80)
print("STEP 1 WINE DATASET TEST COMPLETE")
print("=" * 80)
