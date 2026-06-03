import os
import sys
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Running reproduce_results_offline.py to capture metrics...")
    
    # Run the offline evaluation script
    result = subprocess.run([sys.executable, "scripts/reproduce_results_offline.py"], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error running evaluation script:")
        print(result.stderr)
        return
        
    output = result.stdout
    print("Captured Output:")
    print(output)
    
    # Extract the metrics using robust regex matching the new side-by-side layout
    # Expected format:
    #   Step 1 Accuracy                |                  97.2% |                    50.0%
    #   Step 2 Accuracy                |                  88.9% |                    29.2%
    #   Step 3 Accuracy                |                  88.9% |                    55.6%
    #   Exact Sequence Match Accuracy  |                  87.5% |                    11.1%
    
    step1_match = re.search(r"Step 1 Accuracy\s*\|\s*([\d\.]+)%\s*\|\s*([\d\.]+)%", output)
    step2_match = re.search(r"Step 2 Accuracy\s*\|\s*([\d\.]+)%\s*\|\s*([\d\.]+)%", output)
    step3_match = re.search(r"Step 3 Accuracy\s*\|\s*([\d\.]+)%\s*\|\s*([\d\.]+)%", output)
    exact_match = re.search(r"Exact Sequence Match Accuracy\s*\|\s*([\d\.]+)%\s*\|\s*([\d\.]+)%", output)
    
    if not (step1_match and step2_match and step3_match and exact_match):
        print("Failed to parse metrics from output.")
        return
        
    step1_concat = float(step1_match.group(1))
    step1_blind  = float(step1_match.group(2))
    
    step2_concat = float(step2_match.group(1))
    step2_blind  = float(step2_match.group(2))
    
    step3_concat = float(step3_match.group(1))
    step3_blind  = float(step3_match.group(2))
    
    exact_concat = float(exact_match.group(1))
    exact_blind  = float(exact_match.group(2))
    
    # Set up styling for a clean academic look
    sns.set_theme(style="white")
    
    labels = ['Step 1 Accuracy', 'Step 2 Accuracy', 'Step 3 Accuracy', 'Exact 3-Step Match']
    concat_vals = [step1_concat, step2_concat, step3_concat, exact_concat]
    blind_vals = [step1_blind, step2_blind, step3_blind, exact_blind]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    rects1 = ax.bar(x - width/2, concat_vals, width, label='Concatenated (Validation Sandbox)', color='#2b7bba', edgecolor='#F0F0F0', linewidth=1)
    rects2 = ax.bar(x + width/2, blind_vals, width, label='True Blind (Tracing Limit)', color='#e05c5c', edgecolor='#F0F0F0', linewidth=1)
    
    ax.set_ylim(0, 105)
    ax.set_ylabel('Accuracy (%)\n', fontsize=13, fontweight='bold')
    ax.set_title('Platform-Chain Reconstruction Accuracy Comparison\n(Leak-Free Holdout Set: 72 Chains)\n', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.legend(frameon=True, facecolor='#FCFCFC', edgecolor='#E0E0E0', fontsize=11)
    
    # Add values on top of the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333')
                        
    autolabel(rects1)
    autolabel(rects2)
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    sns.despine()
    plt.tight_layout()
    
    # Ensure assets directory exists
    os.makedirs('assets', exist_ok=True)
    out_path = os.path.join('assets', 'offline_evaluation_metrics.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Chart successfully saved to {out_path}")

if __name__ == "__main__":
    main()
