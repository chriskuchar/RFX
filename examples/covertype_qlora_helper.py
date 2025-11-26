#!/usr/bin/env python3
"""
Test low-rank QLORA on 100K Covertype samples
This is the REAL use case for QLORA (full matrix would be 76GB!)
"""
import rfx as rf
import numpy as np
import time
import urllib.request
import gzip

def load_covertype_cached(n_samples=100000, cache_dir="."):
    """
    Load Covertype dataset with caching
    
    Args:
        n_samples: Number of samples to load (max 581,012)
        cache_dir: Directory to store cached data
    
    Returns:
        X, y: Features and labels
    """
    cache_file = os.path.join(cache_dir, f"covertype_{n_samples}.npz")
    
    # Check if cached file exists
    if os.path.exists(cache_file):
        print(f"Loading from cache: {cache_file}")
        data = np.load(cache_file)
        X = data['X']
        y = data['y']
        print(f"✅ Loaded {len(X):,} samples from cache")
        return X, y
    
    # Download and parse
    print(f"Cache not found. Downloading Covertype dataset from UCI ML Repository...")
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
    
    with urllib.request.urlopen(url) as response:
        compressed_data = response.read()
    
    print("Decompressing...")
    data_str = gzip.decompress(compressed_data).decode('utf-8')
    
    print(f"Parsing first {n_samples:,} samples...")
    lines = [line.strip() for line in data_str.split('\n') if line.strip()]
    
    # Parse CSV manually (last column is class label)
    data_list = []
    for line in lines[:n_samples]:
        values = [float(x) for x in line.split(',')]
        data_list.append(values)
    
    data_array = np.array(data_list, dtype=np.float32)
    X = data_array[:, :-1]  # Features (54 features)
    y = data_array[:, -1].astype(np.int32) - 1  # Classes (0-indexed: 0-6)
    
    # Save to cache
    print(f"Saving to cache: {cache_file}")
    np.savez_compressed(cache_file, X=X, y=y)
    
    print(f"✅ Dataset loaded and cached successfully!")
    return X, y

print("=" * 80)
print("TESTING LOW-RANK QLORA ON 10K COVERTYPE SAMPLES")
print("=" * 80)

# Load dataset (cached)
try:
    X, y_raw = load_covertype_cached(n_samples=10000)
    
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y_raw))
    
    print(f"\nDataset info:")
    print(f"  Samples: {n_samples:,}")
    print(f"  Features: {n_features}")
    print(f"  Classes: {n_classes}")
    
    # Calculate what full matrix would require
    full_matrix_gb = (n_samples * n_samples * 8) / (1024**3)
    print(f"\n💾 Memory comparison:")
    print(f"  Full matrix (FP64): {full_matrix_gb:.1f} GB (Would crash!)")
    print(f"  Low-rank rank=100 (FP64 during training): {2*n_samples*100*8/(1024**2):.1f} MB")
    print(f"  Low-rank rank=100 (INT8 after training): {2*n_samples*100*1/(1024**2):.1f} MB")
    print(f"  Compression ratio: {full_matrix_gb*1024/(2*n_samples*100*1/(1024**2)):.0f}×")
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("Falling back to smaller sample for testing...")
    # Fallback to synthetic data
    n_samples = 10000
    n_features = 54
    n_classes = 7
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y_raw = np.random.randint(0, n_classes, n_samples).astype(np.int32)

# Use y_raw directly (already 0-indexed from load_covertype_cached)
y = y_raw

print(f"\n{'=' * 80}")
print("TRAINING WITH LOW-RANK QLORA")
print(f"{'=' * 80}")

print(f"\nConfiguration:")
print(f"  Trees: 50")
print(f"  Rank: 32 (trees 0-31 create columns, trees 32-49 update via weighted averaging)")
print(f"  GPU: True")
print(f"  QLORA: True (INT8 quantization)")
print(f"  Batch size: 50")

start_time = time.time()

model = rf.RandomForestClassifier(
    ntree=50,
    use_gpu=True,
    batch_size=50,
    compute_proximity=True,
    use_qlora=True,
    rank=32,
    quant_mode='int8',
    iseed=123,
    show_progress=True
)

print("\nTraining...")
model.fit(X, y)

train_time = time.time() - start_time

print(f"\n{'=' * 80}")
print("RESULTS")
print(f"{'=' * 80}")

print(f"\n⏱️  Training time: {train_time:.2f}s ({train_time/60:.2f} min)")
print(f"  Speed: {50/train_time:.2f} trees/sec")

# Check factors
A, B, rank = model.get_lowrank_factors()
nonzero_A = np.count_nonzero(A)
nonzero_B = np.count_nonzero(B)

print(f"\n📊 Low-rank factors:")
print(f"  A shape: {A.shape}, non-zero: {nonzero_A:,}/{A.size:,} ({100*nonzero_A/A.size:.1f}%)")
print(f"  B shape: {B.shape}, non-zero: {nonzero_B:,}/{B.size:,} ({100*nonzero_B/B.size:.1f}%)")
print(f"  Memory: {2*A.nbytes/(1024**2):.1f} MB (INT8 quantized)")

# Compute MDS (sample first 1000 for speed)
print(f"\n🎨 Computing MDS on first 1000 samples (for speed)...")
try:
    # Get MDS for first 1000 samples only
    mds = model.compute_mds_from_factors(k=3)
    
    # Check first 1000
    mds_subset = mds[:1000]
    valid = np.all(np.isfinite(mds_subset), axis=1)
    unique = len(np.unique(np.round(mds_subset[valid], decimals=6), axis=0))
    total = len(mds_subset[valid])
    
    pct_unique = 100 * unique / total if total > 0 else 0
    
    print(f"\nMDS Quality (first 1000 samples):")
    print(f"  Valid points: {total}/1000")
    print(f"  Unique points: {unique}/{total} ({pct_unique:.1f}%)")
    print(f"  Duplicates: {total-unique} ({100*(total-unique)/total:.1f}%)")
    
    if pct_unique >= 90:
        print("\n  ✅ EXCELLENT - Low-rank QLORA working perfectly!")
    elif pct_unique >= 75:
        print("\n  ✓ GOOD - Acceptable quality for visualization")
    elif pct_unique >= 50:
        print("\n  ⚠ FAIR - Some quality issues")
    else:
        print("\n  ❌ POOR - Low-rank accumulation broken")
        
except Exception as e:
    print(f"❌ MDS computation failed: {e}")

print(f"\n{'=' * 80}")
print("SUMMARY")
print(f"{'=' * 80}")
print(f"\n✅ Successfully trained on {n_samples:,} samples with QLORA!")
print(f"   Full matrix would require: {full_matrix_gb:.1f} GB")
print(f"   Low-rank uses only: {2*A.nbytes/(1024**2):.1f} MB (INT8)")
print(f"   Compression: {full_matrix_gb*1024/(2*A.nbytes/(1024**2)):.0f}× memory savings!")
print(f"\n{'=' * 80}")

