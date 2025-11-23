#!/usr/bin/env python3
"""
Test GPU classification on wine dataset and create rfviz with lowrank 3D MDS
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.abspath('.'), 'python'))

import urllib.request
import pandas as pd
import numpy as np
import io
import RFX as rf

# Load wine dataset
print("Loading wine dataset...")
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
with urllib.request.urlopen(url) as response:
    data = response.read().decode('utf-8')
df = pd.read_csv(io.StringIO(data), delimiter=';')
X = df.iloc[:, :-1].values.astype(np.float32)
y_original = df.iloc[:, -1].values.astype(np.int32)
feature_names = df.columns[:-1].tolist()

# Convert to 0-based classes
unique_classes = np.unique(y_original)
class_map = {cls: idx for idx, cls in enumerate(unique_classes)}
y = np.array([class_map[cls] for cls in y_original], dtype=np.int32)

print(f'Loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(unique_classes)} classes')
print(f'Classes: {np.unique(y)}')

# Train GPU model with lowrank proximity
# Use smaller dataset for testing
X_small = X[:500]
y_small = y[:500]
print(f'\nUsing {X_small.shape[0]} samples for GPU training...')

print('\nTraining GPU model (NF4 lowrank proximity)...')
model = rf.RandomForestClassifier(
    ntree=50, mtry=3, nsample=X_small.shape[0], nclass=len(unique_classes),
    use_gpu=True, batch_size=50, iseed=42,
    compute_proximity=True, compute_local_importance=True,
    use_casewise=True, use_qlora=True, rank=50
)
model.fit(X_small, y_small)
print('Model trained!')

# Check lowrank factors
A, B, rank = model.get_lowrank_factors()
print(f'\nLow-rank factors: A.shape={A.shape}, B.shape={B.shape}, rank={rank}')
print(f'A sum: {A.sum():.6f}, B sum: {B.sum():.6f}')
print(f'A == B? {np.allclose(A, B, atol=1e-6)}')
print(f'A - B max diff: {np.abs(A - B).max():.6f}')

# Reconstruct proximity to check diagonal
gpu_prox_recon = A @ B.T
print(f'\nReconstructed proximity: shape={gpu_prox_recon.shape}')
print(f'Diagonal: min={np.diag(gpu_prox_recon).min():.6f}, '
      f'max={np.diag(gpu_prox_recon).max():.6f}, mean={np.diag(gpu_prox_recon).mean():.6f}')

# Get OOB error
oob_err = model.get_oob_error()
print(f'\nOOB error: {oob_err:.4f}')

# Create rfviz with lowrank 3D MDS
print('\nCreating rfviz visualization with lowrank 3D MDS...')
try:
    from create_rfviz_2x2 import create_2x2_grid_html
    
    # Create a wrapper that uses lowrank MDS instead of full proximity
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Get local importance
    local_imp = model.get_local_importance()
    
    # Get OOB predictions
    try:
        oob_pred = model.get_oob_predictions()
    except:
        oob_pred = model.predict(X_small)
    
    # Get 3D MDS from lowrank factors
    print('Computing 3D MDS from lowrank factors...')
    mds_coords = model.compute_mds_3d_from_factors()
    if mds_coords is None or len(mds_coords) == 0:
        print('Warning: Could not compute MDS from factors, using fallback')
        mds_coords = None
    else:
        mds_3d = np.array(mds_coords).reshape(-1, 3)
        print(f'MDS coordinates: shape={mds_3d.shape}')
    
    # Create 2x2 subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Input Features Parallel Coordinates',
                       'Local Importance Parallel Coordinates',
                       '3D MDS from Low-Rank Factors (NF4)',
                       'Class Votes Heatmap'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter3d"}, {"type": "heatmap"}]],
        horizontal_spacing=0.12,
        vertical_spacing=0.15
    )
    
    # Top-left: Input features parallel coordinates (first 5 features)
    n_features_show = min(5, X_small.shape[1])
    for i in range(X_small.shape[0]):
        fig.add_trace(
            go.Scatter(x=list(range(n_features_show)), 
                      y=X[i, :n_features_show],
                      mode='lines+markers', line=dict(width=1),
                      marker=dict(size=3, color=y[i], colorscale='Viridis',
                                showscale=False),
                      showlegend=False, name=f'Sample {i}'),
            row=1, col=1
        )
    
    # Top-right: Local importance parallel coordinates
    if local_imp is not None:
        n_features_show = min(5, local_imp.shape[1])
        for i in range(X_small.shape[0]):
            fig.add_trace(
                go.Scatter(x=list(range(n_features_show)),
                          y=local_imp[i, :n_features_show],
                          mode='lines+markers', line=dict(width=1),
                          marker=dict(size=3, color=y[i], colorscale='Viridis',
                                    showscale=False),
                          showlegend=False),
                row=1, col=2
            )
    
    # Bottom-left: 3D MDS from lowrank factors
    if mds_coords is not None:
        fig.add_trace(
            go.Scatter3d(x=mds_3d[:, 0], y=mds_3d[:, 1], z=mds_3d[:, 2],
                        mode='markers',
                        marker=dict(size=5, color=y, colorscale='Viridis',
                                  showscale=True),
                        name='MDS 3D (Low-Rank)'),
            row=2, col=1
        )
    else:
        # Fallback: use reconstructed proximity for MDS
        try:
            prox = model.get_proximity_matrix()
            prox = np.array(prox).reshape(X_small.shape[0], X_small.shape[0])
            from sklearn.manifold import MDS
            mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
            # Convert proximity to distance
            dist = 1.0 - prox
            dist = (dist + dist.T) / 2  # Make symmetric
            mds_3d = mds.fit_transform(dist)
            fig.add_trace(
                go.Scatter3d(x=mds_3d[:, 0], y=mds_3d[:, 1], z=mds_3d[:, 2],
                            mode='markers',
                            marker=dict(size=5, color=y, colorscale='Viridis',
                                      showscale=True),
                            name='MDS 3D (Fallback)'),
                row=2, col=1
            )
        except Exception as e:
            print(f'Warning: Could not create MDS plot: {e}')
    
    # Bottom-right: Class votes heatmap
    unique_labels = np.unique(y_small)
    n_classes = len(unique_labels)
    vote_matrix = np.zeros((X_small.shape[0], n_classes))
    for i in range(X_small.shape[0]):
        pred_class = oob_pred[i]
        if pred_class in unique_labels:
            class_idx = np.where(unique_labels == pred_class)[0][0]
            vote_matrix[i, class_idx] = 1.0
    
    fig.add_trace(
        go.Heatmap(z=vote_matrix, colorscale='Viridis',
                  name='Votes', showscale=True),
        row=2, col=2
    )
    
    fig.update_layout(
        title='Wine Quality Dataset - GPU Random Forest Visualization (NF4 Low-Rank)',
        height=1000,
        showlegend=False
    )
    
    output_file = 'rfviz_wine_gpu_nf4.html'
    fig.write_html(output_file)
    print(f'\n✓ Saved visualization to {output_file}')
    print('   Layout: Parallel Coords (Input) | Parallel Coords (Local Imp)')
    print('           3D MDS (Low-Rank NF4) | Class Votes Heatmap')
    
except Exception as e:
    print(f'Error creating rfviz: {e}')
    import traceback
    traceback.print_exc()

