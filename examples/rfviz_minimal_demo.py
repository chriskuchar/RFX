#!/usr/bin/env python3
"""Minimal test: GPU model + RFViz HTML generation with proper 3D low-rank"""

import numpy as np
import RFX as rf

print("=" * 60)
print("MINIMAL TEST: GPU + RFViz with 3D Low-Rank")
print("=" * 60)

# Load data
X, y = rf.load_wine()
n_samples, n_features = X.shape
n_classes = len(np.unique(y))
print(f"\nData: {n_samples} samples, {n_features} features, {n_classes} classes")

# Feature names
FEATURE_NAMES = [
    'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
    'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
    'Proanthocyanins', 'Color intensity', 'Hue',
    'OD280/OD315 of diluted wines', 'Proline'
]

# Train GPU model with low-rank proximity (rank=32 for good approximation)
print("\nTraining GPU model with 100 trees, INT8 proximity (rank=32)...")
model = rf.RandomForestClassifier(
    ntree=100,
    mtry=4,
    nsample=n_samples,
    nclass=n_classes,
    use_gpu=True,
    batch_size=10,
    iseed=42,
    compute_proximity=True,
    compute_importance=True,
    compute_local_importance=True,
    use_casewise=True,
    use_qlora=True,
    quant_mode='int8',  # INT8 now fixed!
    rank=32  # Use rank=32 for better approximation, MDS will extract 3D
)
model.fit(X, y)

print(f"OOB Error: {model.get_oob_error():.4f}")

# Get low-rank factors
A, B, rank = model.get_lowrank_factors()
print(f"\nLow-rank factors:")
print(f"   A.shape: {A.shape}")
print(f"   B.shape: {B.shape}")
print(f"   Actual rank: {rank}")

if rank == 32:
    print(f"\nCORRECT: Rank is 32 (as requested)")
else:
    print(f"\nNOTE: Rank is {rank}")

# Show first 3 columns as x,y,z
print(f"\nLow-rank factor values (first 5 samples, first 3 dims):")
print(f"   {'Sample':<8} {'Dim0':>12} {'Dim1':>12} {'Dim2':>12}")
print(f"   {'-'*48}")
for i in range(min(5, n_samples)):
    d0 = A[i,0] if A.shape[1] > 0 else 0
    d1 = A[i,1] if A.shape[1] > 1 else 0
    d2 = A[i,2] if A.shape[1] > 2 else 0
    print(f"   {i:<8} {d0:>12.4f} {d1:>12.4f} {d2:>12.4f}")

# Memory comparison
full_mem = n_samples**2 * 4 / (1024**2)
lr_mem = 2 * n_samples * rank * 4 / (1024**2)
print(f"\nMemory:")
print(f"   Full matrix: {full_mem:.4f} MB")
print(f"   Low-rank: {lr_mem:.4f} MB")
print(f"   Compression: {full_mem/lr_mem:.1f}x")

# Compute MDS from factors
print("\nComputing 3D MDS from low-rank factors...")
try:
    mds_3d = model.compute_mds_from_factors(k=3)
    print(f"   MDS shape: {mds_3d.shape}")
    
    # Count valid vs invalid points
    valid_mask = np.all(np.isfinite(mds_3d), axis=1)
    n_valid = np.sum(valid_mask)
    n_invalid = n_samples - n_valid
    print(f"\n   POINT COUNT:")
    print(f"   Total samples: {n_samples}")
    print(f"   Valid MDS points (finite): {n_valid}")
    print(f"   Invalid MDS points (NaN/inf): {n_invalid}")
    
    if n_invalid > 0:
        print(f"\n   WARNING: {n_invalid} points have NaN/inf coordinates!")
        # Show which samples are invalid
        invalid_indices = np.where(~valid_mask)[0]
        print(f"   Invalid sample indices: {invalid_indices[:20]}{'...' if len(invalid_indices) > 20 else ''}")
    
    print(f"\n   MDS coordinates (first 5 samples):")
    print(f"   {'Sample':<8} {'X':>12} {'Y':>12} {'Z':>12}")
    print(f"   {'-'*48}")
    for i in range(min(5, n_samples)):
        print(f"   {i:<8} {mds_3d[i,0]:>12.4f} {mds_3d[i,1]:>12.4f} {mds_3d[i,2]:>12.4f}")
    
    # Stats per dimension
    print(f"\n   MDS dimension stats:")
    for dim, name in enumerate(['X', 'Y', 'Z']):
        col = mds_3d[:, dim]
        valid_col = col[np.isfinite(col)]
        if len(valid_col) > 0:
            print(f"   {name}: min={valid_col.min():.4f}, max={valid_col.max():.4f}, range={valid_col.max()-valid_col.min():.4f}")
        else:
            print(f"   {name}: ALL NaN/inf!")
            
except Exception as e:
    print(f"   MDS failed: {e}")

# Generate RFViz
print("\nGenerating RFViz HTML...")
try:
    fig = rf.rfviz(
        rf_model=model,
        X=X,
        y=y,
        feature_names=FEATURE_NAMES,
        n_clusters=3,
        title="Wine - GPU Low-Rank INT8 (Rank=32, 100 Trees)",
        output_file="rfviz_int8_fixed.html",
        show_in_browser=False,
        save_html=True,
        mds_k=3
    )
    print(f"RFViz HTML generated successfully!")
    print(f"   Output: rfviz_int8_fixed.html")
except Exception as e:
    print(f"RFViz failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
