#!/usr/bin/env python3
"""
Wine Classification - Basic Example
====================================
Complete example showing:
- OOB error and confusion matrix
- Validation set predictions and metrics
- Overall and local feature importance
- Visualization plots

This reproduces the "Basic Classification" example from the README.
"""

import os
import numpy as np
import RFX as rf

# Optional: matplotlib for plots
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not found. Skipping plot generation.")

# Feature names for Wine dataset
FEATURE_NAMES = [
    'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
    'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
    'Proanthocyanins', 'Color intensity', 'Hue',
    'OD280/OD315 of diluted wines', 'Proline'
]

print("=" * 80)
print("WINE CLASSIFICATION - BASIC EXAMPLE")
print("=" * 80)

# Load Wine dataset (built-in)
print("\n📊 Loading Wine dataset...")
X, y = rf.load_wine()

# Simple train/validation split (80/20)
np.random.seed(123)
indices = np.random.permutation(len(X))
n_train = int(0.8 * len(X))
train_idx, val_idx = indices[:n_train], indices[n_train:]
X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

print(f"  Total samples: {len(X)}")
print(f"  Training: {len(X_train)}, Validation: {len(X_val)}")
print(f"  Features: {X.shape[1]}")
print(f"  Classes: {len(np.unique(y))}")

# Create and train model
print("\n🌲 Training Random Forest...")
model = rf.RandomForestClassifier(
    ntree=100,
    compute_importance=True,
    compute_local_importance=True,
    compute_proximity=True,  # Enable for rfviz
    use_gpu=False  # Set to True for GPU acceleration
)

model.fit(X_train, y_train)
print("✓ Training completed")

print("\n💡 Note: All features (importance, local importance, proximity) are available")
print("   with GPU acceleration too! Just set use_gpu=True and optionally use_qlora=True")
print("   for memory-efficient proximity computation on large datasets.")

# ==============================================================================
# OUT-OF-BAG (OOB) EVALUATION
# ==============================================================================
print("\n" + "=" * 80)
print("OUT-OF-BAG (OOB) EVALUATION")
print("=" * 80)

# Get OOB error (unbiased generalization estimate)
oob_error = model.get_oob_error()
print(f"\nOOB Error: {oob_error:.4f}")
print(f"OOB Accuracy: {1 - oob_error:.4f}")

# Get OOB confusion matrix
oob_pred = model.get_oob_predictions()
confusion = rf.confusion_matrix(y_train, oob_pred)
print("\nOOB Confusion Matrix:")
print("       Pred 0  Pred 1  Pred 2")
print("     " + "-" * 26)
for i in range(3):
    print(f"True {i} | {confusion[i, 0]:4d}   {confusion[i, 1]:4d}   {confusion[i, 2]:4d}")

# ==============================================================================
# VALIDATION SET EVALUATION
# ==============================================================================
print("\n" + "=" * 80)
print("VALIDATION SET EVALUATION")
print("=" * 80)

# Predict on validation set
y_pred = model.predict(X_val)
val_accuracy = np.sum(y_val == y_pred) / len(y_val)
print(f"\nValidation Accuracy: {val_accuracy:.4f}")

# Classification report (built-in RFX function)
print(rf.classification_report(y_val, y_pred))

# ==============================================================================
# FEATURE IMPORTANCE
# ==============================================================================
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE")
print("=" * 80)

# Overall feature importance
importance = model.feature_importances_()
top_indices = np.argsort(importance)[-3:][::-1]

print("\n📊 Top 3 Most Important Features:")
for rank, idx in enumerate(top_indices, 1):
    print(f"  {rank}. {FEATURE_NAMES[idx]:<35s} (score: {importance[idx]:.2f})")

# Local feature importance for first sample
local_imp = model.get_local_importance()
sample_idx = 0
sample_imp = local_imp[sample_idx]

print(f"\n📊 Local Importance for Sample {sample_idx}:")
print(f"   True class: {y_train[sample_idx]}, Predicted: {model.predict(X_train[sample_idx:sample_idx+1])[0]}")
sorted_local_idx = np.argsort(np.abs(sample_imp))[::-1][:5]
print("\n   Top 5 features for this sample:")
for rank, idx in enumerate(sorted_local_idx, 1):
    impact = "helpful" if sample_imp[idx] > 0 else "misleading"
    print(f"   {rank}. {FEATURE_NAMES[idx]:<35s} {sample_imp[idx]:+.4f} ({impact})")

# ==============================================================================
# GENERATE PLOTS (if matplotlib available)
# ==============================================================================
if HAS_MATPLOTLIB:
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    
    # Overall importance plot
    print("\n📈 Generating overall importance plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    sorted_idx = np.argsort(importance)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance)))
    bars = ax.barh(range(len(importance)), importance[sorted_idx], color=colors)
    ax.set_yticks(range(len(importance)))
    ax.set_yticklabels([FEATURE_NAMES[i] for i in sorted_idx])
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title('Overall Feature Importance\nWine Dataset (100 Trees)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for bar, idx in zip(bars, sorted_idx):
        width = bar.get_width()
        ax.text(width + 0.2, bar.get_y() + bar.get_height()/2, 
                f'{importance[idx]:.2f}',
                ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'overall_importance.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved to {output_path}")
    plt.close()
    
    # Local importance plot
    print("\n📈 Generating local importance plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    sorted_idx = np.argsort(np.abs(sample_imp))
    colors = ['#2ecc71' if sample_imp[i] > 0 else '#e74c3c' for i in sorted_idx]
    bars = ax.barh(range(len(sample_imp)), sample_imp[sorted_idx], color=colors, 
                   edgecolor='black', linewidth=0.8)
    ax.set_yticks(range(len(sample_imp)))
    ax.set_yticklabels([FEATURE_NAMES[i] for i in sorted_idx])
    ax.set_xlabel('Local Importance Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title(f'Local Feature Importance for Sample {sample_idx}\n(True Class: {y_train[sample_idx]}, Predicted: {model.predict(X_train[sample_idx:sample_idx+1])[0]})', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Helpful (positive impact)'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Misleading (negative impact)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, frameon=True)
    
    # Add value labels
    for bar, idx in zip(bars, sorted_idx):
        width = bar.get_width()
        label_x = width + (0.003 if width > 0 else -0.003)
        ha = 'left' if width > 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{sample_imp[idx]:.3f}',
                ha=ha, va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'local_importance.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved to {output_path}")
    plt.close()
    
    # RFviz interactive visualization
    print("\n📈 Generating RFviz interactive visualization...")
    output_path = os.path.join(os.path.dirname(__file__), 'rfviz_example.html')
    rf.rfviz(
        rf_model=model,
        X=X_train,
        y=y_train,
        feature_names=FEATURE_NAMES,
        output_file=output_path,
        show_in_browser=False
    )
    print(f"   ✓ Saved interactive HTML to {output_path}")
    print(f"   💡 Open this file in a browser to explore the 2×2 grid visualization:")
    print(f"      - Input features (parallel coordinates)")
    print(f"      - Local importance (parallel coordinates)")
    print(f"      - 3D MDS proximity plot (rotatable)")
    print(f"      - Class votes heatmap")
    print(f"      All with linked brushing - select samples in one plot to highlight in all!")
    print(f"\n   📸 To create a static screenshot (rfviz_example.png):")
    print(f"      Open the HTML in a browser and take a screenshot of the 2×2 grid.")

print("\n" + "=" * 80)
print("EXAMPLE COMPLETE")
print("=" * 80)
print("\n✨ This example demonstrates:")
print("   - OOB error estimation (unbiased without separate test set)")
print("   - Confusion matrix analysis")
print("   - Validation set predictions")
print("   - Overall feature importance (global explanations)")
print("   - Local feature importance (instance-level explanations)")
print("   - Proximity matrices for sample similarity analysis")
if HAS_MATPLOTLIB:
    print("   - Visualization plots saved to examples/")
    print("   - Interactive RFviz HTML with linked brushing")
print("\n💡 Next steps:")
print("   - Try with GPU acceleration: use_gpu=True")
print("   - Use QLORA for memory-efficient proximity: use_qlora=True, quant_mode='int8'")
print("   - Enable case-wise mode: use_casewise=True")
print("   - See examples/ folder for more advanced use cases")

