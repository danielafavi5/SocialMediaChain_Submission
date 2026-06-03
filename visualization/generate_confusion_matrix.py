import os
import sys
import subprocess
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def main():
    print("Running reproduce_results_offline.py to capture evaluation outputs...")
    
    # Run the offline evaluation script
    result = subprocess.run([sys.executable, "scripts/reproduce_results_offline.py"], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error running evaluation script:")
        print(result.stderr)
        return
        
    output = result.stdout
    print("Successfully captured evaluation output. Parsing sequences...")

    # Parse true and predicted sequences using a highly precise regular expression
    # Row pattern:   01804966_177151723 telegram>slack>discord         telegram>discord>slack    [--]
    pattern = re.compile(r"^\s*([a-zA-Z0-9_]+)\s+([a-z_>]+)\s+([a-z_>]+)\s+\[(?:OK|--)\]")
    
    true_sequences = []
    pred_sequences = []
    
    for line in output.splitlines():
        match = pattern.match(line)
        if match:
            true_seq = match.group(2)
            pred_seq = match.group(3)
            true_sequences.append(true_seq)
            pred_sequences.append(pred_seq)
            
    if not true_sequences:
        print("Error: Could not parse any sequences from the evaluation output. Please check the evaluation logs.")
        return
        
    print(f"Parsed {len(true_sequences)} evaluated sequence chains.")
    
    # Identify unique sequences present in the output
    unique_sequences = sorted(list(set(true_sequences + pred_sequences)))
    print(f"Unique sequence classes detected ({len(unique_sequences)}): {unique_sequences}")
    
    # Compute the raw confusion matrix
    cm = confusion_matrix(true_sequences, pred_sequences, labels=unique_sequences)
    
    # Normalize by true label count (row normalization)
    with np.errstate(all='ignore'):
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.where(row_sums > 0, cm.astype('float') / row_sums, 0.0)

    # Prepare custom text annotations for each cell
    annot_matrix = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_normalized[i, j]
            annot_matrix[i, j] = f"{pct:.1%}\n(n={count})"
            
    # Set up matplotlib style for a clean, presentation-ready look
    sns.set_theme(style="white")
    plt.figure(figsize=(12, 10))
    
    # Create the heatmap using a blue-teal color palette
    ax = sns.heatmap(
        cm_normalized,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        xticklabels=unique_sequences,
        yticklabels=unique_sequences,
        vmin=0.0,
        vmax=1.0,
        cbar_kws={'label': 'Normalized Proportion (Accuracy)'},
        linewidths=1.5,
        linecolor='#F0F0F0',
        annot_kws={"size": 11, "weight": "bold"}
    )
    
    # Customize titles and labels
    plt.title("Sequence-Level Confusion Matrix (Normalized by True Labels)\n", fontsize=16, fontweight='bold', pad=10)
    plt.xlabel("\nPredicted Sequence (Output)", fontsize=13, fontweight='bold')
    plt.ylabel("True Sequence (Ground Truth)\n", fontsize=13, fontweight='bold')
    
    # Rotate axis labels for readability
    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    # Tight layout to avoid cutting off labels
    plt.tight_layout()
    
    # Ensure assets directory exists and save
    os.makedirs('assets', exist_ok=True)
    out_path = os.path.join('assets', 'sequence_confusion_matrix.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    
    print(f"\nConfusion matrix successfully generated and saved to: {out_path}")

if __name__ == "__main__":
    main()
