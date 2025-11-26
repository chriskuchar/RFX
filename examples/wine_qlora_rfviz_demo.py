#!/usr/bin/env python3
"""
Wine Classification - Low-Rank Proximity + RFViz Inline HTML
Demonstrates QLoRA low-rank proximity and interactive visualization
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

def main():
    print("=" * 70)
    print("  WINE CLASSIFICATION - LOW-RANK PROXIMITY + RFVIZ")
    print("=" * 70)
    
    # Load Wine dataset (built-in)
    X, y = rf.load_wine()
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    
    print(f"\nDataset: Wine (UCI ML - built-in)")
    print(f"   Samples: {n_samples}")
    print(f"   Features: {n_features}")
    print(f"   Classes: {n_classes}")
    print(f"   Class distribution: {np.bincount(y).tolist()}")
    
    ntree = 100
    rank = 32
    
    # =========================================================================
    # 1. GPU LOW-RANK PROXIMITY (QLoRA NF4)
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  1. GPU LOW-RANK PROXIMITY (QLoRA NF4, Rank-{rank})")
    print(f"{'='*70}")
    
    model_lowrank = rf.RandomForestClassifier(
        ntree=ntree,
        mtry=4,
        nsample=n_samples,
        nclass=n_classes,
        use_gpu=True,
        batch_size=0,  # Auto SM-aware batching
        iseed=42,
        compute_proximity=True,
        compute_importance=True,
        compute_local_importance=True,
        use_casewise=True,
        use_qlora=True,      # Enable QLoRA compression
        quant_mode='nf4',    # NF4 quantization
        rank=rank            # Low-rank factorization rank
    )
    
    print(f"\nTraining with QLoRA low-rank proximity...")
    start_time = time.time()
    model_lowrank.fit(X, y)
    elapsed_lowrank = time.time() - start_time
    
    print(f"Training time: {elapsed_lowrank:.2f}s ({ntree/elapsed_lowrank:.1f} trees/sec)")
    
    # Get low-rank factors
    A, B, actual_rank = model_lowrank.get_lowrank_factors()
    print(f"\nLow-Rank Factors:")
    print(f"   A.shape: {A.shape}")
    print(f"   B.shape: {B.shape}")
    print(f"   Rank: {actual_rank}")
    
    # Reconstruct proximity for analysis
    prox_reconstructed = A @ B.T
    print(f"\nReconstructed Proximity Matrix:")
    print(f"   Shape: {prox_reconstructed.shape}")
    print(f"   Diagonal: min={np.diag(prox_reconstructed).min():.4f}, "
          f"max={np.diag(prox_reconstructed).max():.4f}, mean={np.diag(prox_reconstructed).mean():.4f}")
    print(f"   Off-diagonal mean: {prox_reconstructed[np.triu_indices(n_samples, k=1)].mean():.4f}")
    
    # Memory comparison
    full_memory_mb = n_samples**2 * 4 / (1024**2)
    lowrank_memory_mb = 2 * n_samples * actual_rank * 4 / (1024**2)
    compression_ratio = full_memory_mb / lowrank_memory_mb
    
    print(f"\nMemory Efficiency:")
    print(f"   Full proximity matrix: {full_memory_mb:.4f} MB")
    print(f"   Low-rank factors: {lowrank_memory_mb:.4f} MB")
    print(f"   Compression ratio: {compression_ratio:.1f}x")
    
    # OOB and importance
    oob_err_lowrank = model_lowrank.get_oob_error()
    oob_preds_lowrank = model_lowrank.get_oob_predictions()
    importance_lowrank = model_lowrank.feature_importances_()
    local_imp_lowrank = model_lowrank.get_local_importance()
    
    print(f"\nModel Performance:")
    print(f"   OOB Error: {oob_err_lowrank:.4f} ({oob_err_lowrank*100:.2f}%)")
    print(f"   OOB Accuracy: {(1-oob_err_lowrank)*100:.2f}%")
    
    # Confusion Matrix & Classification Report (built-in)
    print(f"\nConfusion Matrix (rf.confusion_matrix):")
    print(rf.confusion_matrix(y.astype(np.int32), oob_preds_lowrank.astype(np.int32)))
    print(f"\nClassification Report (rf.classification_report):")
    print(rf.classification_report(y.astype(np.int32), oob_preds_lowrank.astype(np.int32)))
    
    print(f"\nTop 5 Feature Importance:")
    sorted_idx = np.argsort(importance_lowrank)[::-1]
    for rank_i, idx in enumerate(sorted_idx[:5], 1):
        print(f"   {rank_i}. {FEATURE_NAMES[idx]}: {importance_lowrank[idx]:.6f}")
    
    # Compute MDS from low-rank factors (built-in)
    print(f"\nComputing 3D MDS from low-rank factors...")
    mds_3d_lowrank = model_lowrank.compute_mds_3d_from_factors()
    print(f"   MDS shape: {mds_3d_lowrank.shape}")
    print(f"   MDS range: x=[{mds_3d_lowrank[:, 0].min():.4f}, {mds_3d_lowrank[:, 0].max():.4f}]")
    
    # =========================================================================
    # 2. CPU FULL PROXIMITY (for comparison)
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  2. CPU FULL PROXIMITY (for comparison)")
    print(f"{'='*70}")
    
    model_full = rf.RandomForestClassifier(
        ntree=ntree,
        mtry=4,
        nsample=n_samples,
        nclass=n_classes,
        use_gpu=False,  # CPU
        iseed=42,
        compute_proximity=True,
        compute_importance=True,
        compute_local_importance=True,
        use_casewise=True,
        use_qlora=False  # Full dense matrix
    )
    
    print(f"\nTraining with full proximity matrix (CPU)...")
    start_time = time.time()
    model_full.fit(X, y)
    elapsed_full = time.time() - start_time
    
    print(f"Training time: {elapsed_full:.2f}s ({ntree/elapsed_full:.1f} trees/sec)")
    
    # Get full proximity
    prox_full = model_full.get_proximity_matrix()
    prox_full = np.array(prox_full).reshape(n_samples, n_samples)
    
    print(f"\nFull Proximity Matrix:")
    print(f"   Shape: {prox_full.shape}")
    print(f"   Diagonal: min={np.diag(prox_full).min():.4f}, "
          f"max={np.diag(prox_full).max():.4f}, mean={np.diag(prox_full).mean():.4f}")
    
    oob_err_full = model_full.get_oob_error()
    oob_preds_full = model_full.get_oob_predictions()
    importance_full = model_full.feature_importances_()
    
    print(f"\nModel Performance:")
    print(f"   OOB Error: {oob_err_full:.4f} ({oob_err_full*100:.2f}%)")
    
    # Confusion Matrix & Classification Report (built-in)
    print(f"\nConfusion Matrix (rf.confusion_matrix):")
    print(rf.confusion_matrix(y.astype(np.int32), oob_preds_full.astype(np.int32)))
    print(f"\nClassification Report (rf.classification_report):")
    print(rf.classification_report(y.astype(np.int32), oob_preds_full.astype(np.int32)))
    
    # Compute MDS from full proximity (built-in CPU)
    print(f"\nComputing 3D MDS from full proximity (CPU)...")
    mds_3d_full = model_full.compute_mds_3d_cpu()
    print(f"   MDS shape: {mds_3d_full.shape}")
    
    # =========================================================================
    # 3. COMPARISON
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  3. LOW-RANK vs FULL COMPARISON")
    print(f"{'='*70}")
    
    # Proximity correlation
    upper_tri_idx = np.triu_indices(n_samples, k=1)
    prox_lowrank_flat = prox_reconstructed[upper_tri_idx]
    prox_full_flat = prox_full[upper_tri_idx]
    
    corr_spearman, pval = spearmanr(prox_lowrank_flat, prox_full_flat)
    corr_pearson = np.corrcoef(prox_lowrank_flat, prox_full_flat)[0, 1]
    mae = np.mean(np.abs(prox_lowrank_flat - prox_full_flat))
    
    print(f"\nProximity Matrix Quality:")
    print(f"   Spearman correlation: {corr_spearman:.6f}")
    print(f"   Pearson correlation: {corr_pearson:.6f}")
    print(f"   Mean absolute error: {mae:.6f}")
    
    # MDS correlation
    mds_corr = np.corrcoef(mds_3d_lowrank.flatten(), mds_3d_full.flatten())[0, 1]
    print(f"\nMDS Embedding Quality:")
    print(f"   Correlation (low-rank vs full): {mds_corr:.6f}")
    
    # Importance correlation
    imp_corr, _ = spearmanr(importance_lowrank, importance_full)
    print(f"\nFeature Importance Correlation: {imp_corr:.6f}")
    
    print(f"\nOOB Error Comparison:")
    print(f"   Low-rank: {oob_err_lowrank:.4f} ({oob_err_lowrank*100:.2f}%)")
    print(f"   Full:     {oob_err_full:.4f} ({oob_err_full*100:.2f}%)")
    print(f"   Diff:     {abs(oob_err_lowrank - oob_err_full):.6f}")
    
    if corr_spearman > 0.99:
        print(f"\nEXCELLENT: Low-rank preserves >99% proximity structure!")
    elif corr_spearman > 0.95:
        print(f"\nGOOD: Low-rank preserves >95% proximity structure")
    else:
        print(f"\nWARNING: Low-rank correlation <95%")
    
    # =========================================================================
    # 4. RFVIZ VISUALIZATION
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  4. RFVIZ INTERACTIVE VISUALIZATION")
    print(f"{'='*70}")
    
    # Generate RFViz for low-rank model
    print(f"\nGenerating RFViz for Low-Rank Model...")
    output_file_lowrank = "rfviz_wine_lowrank.html"
    
    try:
        fig_lowrank = rf.rfviz(
            rf_model=model_lowrank,
            X=X,
            y=y,
            feature_names=FEATURE_NAMES,
            n_clusters=3,
            title=f"Wine Classification - Low-Rank Proximity (QLoRA NF4, Rank-{rank})<br>"
                  f"OOB Error: {oob_err_lowrank:.4f} | Compression: {compression_ratio:.1f}x",
            output_file=output_file_lowrank,
            show_in_browser=True,  # Open in browser
            save_html=True,
            mds_k=3
        )
        print(f"   Saved: {output_file_lowrank}")
        print(f"   Opening in browser...")
    except Exception as e:
        print(f"   Error generating low-rank RFViz: {e}")
    
    # Generate RFViz for full proximity model
    print(f"\nGenerating RFViz for Full Proximity Model...")
    output_file_full = "rfviz_wine_full.html"
    
    try:
        fig_full = rf.rfviz(
            rf_model=model_full,
            X=X,
            y=y,
            feature_names=FEATURE_NAMES,
            n_clusters=3,
            title=f"Wine Classification - Full Proximity Matrix (CPU)<br>"
                  f"OOB Error: {oob_err_full:.4f}",
            output_file=output_file_full,
            show_in_browser=True,  # Open in browser
            save_html=True,
            mds_k=3
        )
        print(f"   Saved: {output_file_full}")
        print(f"   Opening in browser...")
    except Exception as e:
        print(f"   Error generating full RFViz: {e}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nGenerated Files:")
    print(f"   - {output_file_lowrank} - Low-rank model visualization")
    print(f"   - {output_file_full} - Full proximity model visualization")
    
    print(f"\nRFViz Features:")
    print(f"   - 2x2 grid: Input Features | Local Importance | 3D MDS | Confusion")
    print(f"   - Linked brushing: Select samples in any plot, highlights in all")
    print(f"   - 3D MDS: Drag to rotate, scroll to zoom")
    print(f"   - Save selection: Click 'Save Selected Subset' button")
    print(f"   - Clear selection: Press 'R' or 'Escape'")
    
    print(f"\nPerformance Summary:")
    print(f"   Low-rank: {elapsed_lowrank:.2f}s, OOB={oob_err_lowrank:.4f}, {compression_ratio:.1f}x compression")
    print(f"   Full:     {elapsed_full:.2f}s, OOB={oob_err_full:.4f}, no compression")
    print(f"   Proximity correlation: {corr_spearman:.4f}")
    
    print("\n" + "=" * 70)
    print("  TEST COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    main()

