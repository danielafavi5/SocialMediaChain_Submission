import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Dynamic relative paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
    
SURF_MODEL_PATH = os.path.join(BASE_DIR, "models", "surface_model.joblib")

def main():
    print("Generating Feature Importance Bar Chart...")
    
    if not os.path.exists(SURF_MODEL_PATH):
        print(f"Error: Model not found at {SURF_MODEL_PATH}. Run export_models.py first.")
        return
        
    try:
        model = joblib.load(SURF_MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    if not hasattr(model, "feature_importances_"):
        print("Error: Loaded model does not expose feature_importances_.")
        return
        
    importances = model.feature_importances_
    
    if len(importances) != 272:
        print(f"Warning: Expected 272 features, but model has {len(importances)}. Chart may be inaccurate.")
        
    feature_families = [
        ("DCT AC Histogram Bins", 21),
        ("DCT Coefficient Energy Map", 21),
        ("Intra-block Markov Transitions", 81),
        ("Luminance Q-table", 64),
        ("Luma Q-table Stats", 4),
        ("Chrominance Q-table", 40),
        ("Chroma Q-table Stats", 4),
        ("Metadata Flags", 6),
        ("Structural Stats", 6),
        ("Ghost Peaks", 5),
        ("Q-Table Backtracking", 6),
        ("Benford's Law Analysis", 9),
        ("Container Byte Analysis", 5)
    ]
    
    family_names = []
    family_importances = []
    
    accum_idx = 0
    for name, dims in feature_families:
        end_idx = accum_idx + dims
        # Sum importances for this family (as a percentage)
        fam_imp = np.sum(importances[accum_idx:end_idx]) * 100.0
        family_names.append(name)
        family_importances.append(fam_imp)
        accum_idx = end_idx
        
    # Sort by importance
    sorted_indices = np.argsort(family_importances)
    sorted_names = [family_names[i] for i in sorted_indices]
    sorted_importances = [family_importances[i] for i in sorted_indices]
    
    # Setup styling
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8), facecolor='#FCFCFC')
    ax = plt.gca()
    ax.set_facecolor('#FCFCFC')
    
    # Draw horizontal bars
    bars = plt.barh(sorted_names, sorted_importances, color='#4A90E2', height=0.6, edgecolor='none')
    
    # Title and Labels
    plt.title("Random Forest Feature Importance by Family", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Aggregate Feature Importance (%)", fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel("Forensic Feature Family", fontsize=12, fontweight='bold', labelpad=10)
    
    plt.xlim(0, max(sorted_importances) * 1.15)
    plt.yticks(fontsize=11)
    plt.xticks(fontsize=10)
    
    # Annotate values
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + (max(sorted_importances) * 0.01), 
            bar.get_y() + bar.get_height()/2.0, 
            f'{width:.1f}%', 
            ha='left', 
            va='center', 
            fontsize=11, 
            fontweight='bold',
            color='#2C3E50'
        )
        
    sns.despine(left=True, bottom=True)
    
    plt.tight_layout()
    os.makedirs('assets', exist_ok=True)
    out_path = os.path.join('assets', 'feature_importance.png')
    plt.savefig(out_path, dpi=300, facecolor='#FCFCFC', bbox_inches='tight')
    
    print(f"Feature Importance Chart successfully saved to: {out_path}")

if __name__ == "__main__":
    main()
