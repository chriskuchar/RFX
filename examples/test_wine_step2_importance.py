#!/usr/bin/env python3
"""
Step 2: Wine Classification - Overall and Local Importance
Test configurations:
1. GPU Non-casewise (batch_size=5, 10 trees)
2. GPU Casewise (batch_size=5, 10 trees)
3. CPU Non-casewise (10 trees)
4. CPU Casewise (10 trees)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.abspath('.'), 'python'))

import numpy as np
import RFX as rf
import urllib.request

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

# Common parameters
ntree = 500  # Need many trees for stable importance estimates with 13 features
iseed = 42
mtry = 4  # sqrt(13) ≈ 3.6
batch_size_gpu = 0  # Auto-scaling - let RFX determine optimal batch size

print("=" * 80)
print(f"STEP 2: OVERALL AND LOCAL IMPORTANCE - WINE DATASET ({ntree} TREES)")
print("=" * 80)

# Load Wine dataset
X, y, feature_names = load_wine_cached()

print(f"\nDataset: Wine (UCI ML)")
print(f"  Samples: {X.shape[0]}")
print(f"  Features: {X.shape[1]}")
print(f"  Classes: {len(np.unique(y))}")
print(f"  Class distribution: {np.bincount(y)}")

results = {}

# ================================================================================
# 1. GPU NON-CASEWISE (batch_size=5, 10 trees)
# ================================================================================
print("\n" + "=" * 80)
print(f"1. GPU NON-CASEWISE (batch_size={batch_size_gpu}, {ntree} trees)")
print("=" * 80)

model_gpu_ncw = rf.RandomForestClassifier(
    ntree=ntree,
    mtry=mtry,
    iseed=iseed,
    use_gpu=True,
    batch_size=batch_size_gpu,
    use_casewise=False,
    compute_importance=True,
    compute_local_importance=True
)

print("Training...")
model_gpu_ncw.fit(X, y)

# Get importance
overall_imp_gpu_ncw = model_gpu_ncw.feature_importances_()
local_imp_gpu_ncw = model_gpu_ncw.get_local_importance()

print(f"\nOverall Importance (shape: {overall_imp_gpu_ncw.shape}):")
for i, (feat, imp) in enumerate(zip(feature_names, overall_imp_gpu_ncw)):
    print(f"  {feat:35s}: {imp:8.5f}")

print(f"\nLocal Importance (shape: {local_imp_gpu_ncw.shape}):")
print(f"  Mean per feature: {np.mean(local_imp_gpu_ncw, axis=0)}")
print(f"  Std per feature:  {np.std(local_imp_gpu_ncw, axis=0)}")

results['gpu_ncw'] = {
    'overall': overall_imp_gpu_ncw,
    'local': local_imp_gpu_ncw
}

# ================================================================================
# 2. GPU CASEWISE (batch_size=5, 10 trees)
# ================================================================================
print("\n" + "=" * 80)
print(f"2. GPU CASEWISE (batch_size={batch_size_gpu}, {ntree} trees)")
print("=" * 80)

model_gpu_cw = rf.RandomForestClassifier(
    ntree=ntree,
    mtry=mtry,
    iseed=iseed,
    use_gpu=True,
    batch_size=batch_size_gpu,
    use_casewise=True,
    compute_importance=True,
    compute_local_importance=True
)

print("Training...")
model_gpu_cw.fit(X, y)

# Get importance
overall_imp_gpu_cw = model_gpu_cw.feature_importances_()
local_imp_gpu_cw = model_gpu_cw.get_local_importance()

print(f"\nOverall Importance (shape: {overall_imp_gpu_cw.shape}):")
for i, (feat, imp) in enumerate(zip(feature_names, overall_imp_gpu_cw)):
    print(f"  {feat:35s}: {imp:8.5f}")

print(f"\nLocal Importance (shape: {local_imp_gpu_cw.shape}):")
print(f"  Mean per feature: {np.mean(local_imp_gpu_cw, axis=0)}")
print(f"  Std per feature:  {np.std(local_imp_gpu_cw, axis=0)}")

results['gpu_cw'] = {
    'overall': overall_imp_gpu_cw,
    'local': local_imp_gpu_cw
}

# ================================================================================
# 3. CPU NON-CASEWISE (10 trees)
# ================================================================================
print("\n" + "=" * 80)
print(f"3. CPU NON-CASEWISE ({ntree} trees)")
print("=" * 80)

model_cpu_ncw = rf.RandomForestClassifier(
    ntree=ntree,
    mtry=mtry,
    iseed=iseed,
    use_gpu=False,
    use_casewise=False,
    compute_importance=True,
    compute_local_importance=True
)

print("Training...")
model_cpu_ncw.fit(X, y)

# Get importance
overall_imp_cpu_ncw = model_cpu_ncw.feature_importances_()
local_imp_cpu_ncw = model_cpu_ncw.get_local_importance()

print(f"\nOverall Importance (shape: {overall_imp_cpu_ncw.shape}):")
for i, (feat, imp) in enumerate(zip(feature_names, overall_imp_cpu_ncw)):
    print(f"  {feat:35s}: {imp:8.5f}")

print(f"\nLocal Importance (shape: {local_imp_cpu_ncw.shape}):")
print(f"  Mean per feature: {np.mean(local_imp_cpu_ncw, axis=0)}")
print(f"  Std per feature:  {np.std(local_imp_cpu_ncw, axis=0)}")

results['cpu_ncw'] = {
    'overall': overall_imp_cpu_ncw,
    'local': local_imp_cpu_ncw
}

# ================================================================================
# 4. CPU CASEWISE (10 trees)
# ================================================================================
print("\n" + "=" * 80)
print(f"4. CPU CASEWISE ({ntree} trees)")
print("=" * 80)

model_cpu_cw = rf.RandomForestClassifier(
    ntree=ntree,
    mtry=mtry,
    iseed=iseed,
    use_gpu=False,
    use_casewise=True,
    compute_importance=True,
    compute_local_importance=True
)

print("Training...")
model_cpu_cw.fit(X, y)

# Get importance
overall_imp_cpu_cw = model_cpu_cw.feature_importances_()
local_imp_cpu_cw = model_cpu_cw.get_local_importance()

print(f"\nOverall Importance (shape: {overall_imp_cpu_cw.shape}):")
for i, (feat, imp) in enumerate(zip(feature_names, overall_imp_cpu_cw)):
    print(f"  {feat:35s}: {imp:8.5f}")

print(f"\nLocal Importance (shape: {local_imp_cpu_cw.shape}):")
print(f"  Mean per feature: {np.mean(local_imp_cpu_cw, axis=0)}")
print(f"  Std per feature:  {np.std(local_imp_cpu_cw, axis=0)}")

results['cpu_cw'] = {
    'overall': overall_imp_cpu_cw,
    'local': local_imp_cpu_cw
}

# ================================================================================
# COMPARISON SUMMARY
# ================================================================================
print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

print("\n📊 Overall Importance Correlations:")
from scipy.stats import spearmanr

configs = ['gpu_ncw', 'gpu_cw', 'cpu_ncw', 'cpu_cw']
labels = {
    'gpu_ncw': 'GPU Non-casewise',
    'gpu_cw': 'GPU Casewise',
    'cpu_ncw': 'CPU Non-casewise',
    'cpu_cw': 'CPU Casewise'
}

for i, c1 in enumerate(configs):
    for c2 in configs[i+1:]:
        corr, pval = spearmanr(results[c1]['overall'], results[c2]['overall'])
        print(f"  {labels[c1]:20s} vs {labels[c2]:20s}: {corr:.4f} (p={pval:.4e})")

print("\n📊 Local Importance Mean Correlations:")
for i, c1 in enumerate(configs):
    for c2 in configs[i+1:]:
        mean1 = np.mean(results[c1]['local'], axis=0)
        mean2 = np.mean(results[c2]['local'], axis=0)
        corr, pval = spearmanr(mean1, mean2)
        print(f"  {labels[c1]:20s} vs {labels[c2]:20s}: {corr:.4f} (p={pval:.4e})")

print("\n📊 Top 5 Important Features (by Overall Importance):")
for config in configs:
    imp = results[config]['overall']
    top5_idx = np.argsort(imp)[-5:][::-1]
    print(f"\n  {labels[config]}:")
    for idx in top5_idx:
        print(f"    {feature_names[idx]:35s}: {imp[idx]:8.5f}")

print("\n" + "=" * 80)
print("STEP 2 WINE DATASET TEST COMPLETE")
print("=" * 80)
