#!/usr/bin/env python3
"""
3D MDS Visualization Example

Demonstrates how to create interactive 3D MDS plots from proximity matrices
using RFX's low-rank factors. Shows the effect of tree count on MDS coverage.

This example creates a 3D scatter plot colored by class labels, demonstrating
how more trees lead to better MDS coverage (fewer duplicate/invalid points).
"""

import numpy as np
import rfx as rf

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not installed. Install with: pip install plotly")

def plot_3d_mds(mds_coords, y, title="3D MDS Plot", save_html=None):
    """
    Create interactive 3D MDS plot colored by class labels
    
    Parameters:
    -----------
    mds_coords : ndarray (n_samples, 3)
        3D MDS coordinates
    y : ndarray (n_samples,)
        Class labels for coloring
    title : str
        Plot title
    save_html : str, optional
        Path to save interactive HTML file
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Skipping plot.")
        return
    
    # Filter out invalid (NaN/inf) coordinates
    valid_mask = np.all(np.isfinite(mds_coords), axis=1)
    mds_valid = mds_coords[valid_mask]
    y_valid = y[valid_mask]
    
    n_valid = np.sum(valid_mask)
    n_total = len(y)
    
    print(f"\n3D MDS Visualization:")
    print(f"  Valid points: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
    
    # Create class names
    class_names = ['Class 1', 'Class 2', 'Class 3']
    
    # Create figure with separate traces per class
    fig = go.Figure()
    
    for class_idx in np.unique(y_valid):
        mask = y_valid == class_idx
        fig.add_trace(go.Scatter3d(
            x=mds_valid[mask, 0],
            y=mds_valid[mask, 1],
            z=mds_valid[mask, 2],
            mode='markers',
            name=class_names[int(class_idx)],
            marker=dict(
                size=6,
                opacity=0.8,
                line=dict(width=0.5, color='white')
            ),
            text=[f'Sample {i}<br>Class {int(class_idx)}' 
                  for i in np.where(valid_mask)[0][mask]]
        ))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title='MDS Dimension 1',
            yaxis_title='MDS Dimension 2',
            zaxis_title='MDS Dimension 3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        showlegend=True,
        width=900,
        height=700,
        hovermode='closest'
    )
    
    if save_html:
        fig.write_html(save_html)
        print(f"  Saved: {save_html}")
    else:
        fig.show()


def main():
    """
    Compare MDS quality with different tree counts
    """
    print("=" * 80)
    print("3D MDS VISUALIZATION - Wine Dataset")
    print("=" * 80)
    
    # Load Wine dataset
    X, y = rf.load_wine()
    n_samples, n_features = X.shape
    print(f"\nDataset: Wine (UCI ML Repository)")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Classes: {len(np.unique(y))}")
    
    # Test different tree counts
    tree_counts = [100, 500, 1500]
    
    for ntree in tree_counts:
        print(f"\n{'=' * 80}")
        print(f"Training with {ntree} trees...")
        print(f"{'=' * 80}")
        
        # Train model with GPU low-rank proximity
        model = rf.RandomForestClassifier(
            ntree=ntree,
            mtry=4,
            use_gpu=True,
            use_qlora=True,
            quant_mode='int8',
            rank=32,
            compute_proximity=True,
            compute_importance=True,
            iseed=42,
            batch_size=10,
            show_progress=True
        )
        
        model.fit(X, y)
        
        # Compute 3D MDS from low-rank factors
        print(f"\nComputing 3D MDS from low-rank factors...")
        mds_coords = model.compute_mds_from_factors(k=3)
        
        # Calculate coverage
        valid_mask = np.all(np.isfinite(mds_coords), axis=1)
        n_valid = np.sum(valid_mask)
        coverage = 100 * n_valid / n_samples
        
        print(f"  MDS shape: {mds_coords.shape}")
        print(f"  Valid points: {n_valid}/{n_samples} ({coverage:.1f}%)")
        
        if coverage < 95:
            print(f"  ⚠ Low coverage! Recommend {n_samples * 8} trees for >95% coverage")
        else:
            print(f"  ✓ Good coverage!")
        
        # Create 3D plot
        title = f"3D MDS - Wine Dataset ({ntree} trees, {n_valid}/{n_samples} valid)"
        save_path = f"mds_3d_{ntree}trees.html"
        
        plot_3d_mds(mds_coords, y, title=title, save_html=save_path)
        
        # Show MDS properties
        if n_valid > 10:
            mds_valid = mds_coords[valid_mask]
            means = mds_valid.mean(axis=0)
            variances = mds_valid.var(axis=0)
            
            print(f"\n  MDS Properties:")
            print(f"    Mean (should be ≈0): {means}")
            print(f"    Variance (ordered):  {variances}")
            print(f"    Variance explained:  {100*variances/variances.sum():.1f}%")


if __name__ == "__main__":
    main()

