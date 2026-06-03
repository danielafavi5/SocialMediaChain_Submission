import os
import sys
import subprocess
import re
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Running compare_models.py to capture evaluation results...")
    
    # Run the model comparison script
    result = subprocess.run([sys.executable, "scripts/compare_models.py"], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error running compare_models.py:")
        print(result.stderr)
        return
        
    output = result.stdout
    print("Successfully captured comparison output. Parsing metrics...")
    
    # Use re.DOTALL to match across lines
    orig_match = re.search(r"--- Original Model.*?Exact Sequence Match:\s*([\d\.]+)%", output, re.DOTALL)
    new_match = re.search(r"--- 2026 Model.*?Exact Sequence Match:\s*([\d\.]+)%", output, re.DOTALL)
    
    if not (orig_match and new_match):
        print("Error: Could not parse both model scores from compare_models.py output.")
        print(f"Captured output was:\n{output}")
        return
        
    orig_score = float(orig_match.group(1))
    new_score = float(new_match.group(1))
    
    print(f"Parsed Scores:")
    print(f"  Finalized Pipeline (Original Model): {orig_score}%")
    print(f"  Experimental Model (2026 Model)  : {new_score}%")
    
    # Set up styling for a standard presentation slide
    sns.set_theme(style="white")
    plt.figure(figsize=(9, 6.5))
    
    # Bar details
    labels = ['Finalized Pipeline\n(Original Model)', 'Experimental Model\n(2026 Model)']
    values = [orig_score, new_score]
    colors = ['#2b7bba', '#e05c5c'] # Tech blue for production, warm crimson for experimental
    
    # Plot the bar chart
    bars = plt.bar(labels, values, color=colors, width=0.5, edgecolor='#F0F0F0', linewidth=1)
    
    # Customize layout and axes
    plt.ylim(0, 105)
    plt.ylabel('Exact Sequence Match (%)\n', fontsize=13, fontweight='bold')
    plt.title('Sequence-Level Generalization Performance Comparison\n(Leak-Free Holdout Evaluation)\n', fontsize=15, fontweight='bold')
    
    # Annotate values on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.0, 
            height + 1.5, 
            f'{height:.1f}%', 
            ha='center', 
            va='bottom', 
            fontsize=12, 
            fontweight='bold',
            color='#333333'
        )
        
    # Grid and aesthetic polish
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(fontsize=11, fontweight='bold')
    plt.yticks(fontsize=10)
    
    # Despine top and right borders for a modern look
    sns.despine()
    
    plt.tight_layout()
    
    # Ensure assets directory exists and save the plot
    os.makedirs('assets', exist_ok=True)
    out_path = os.path.join('assets', 'model_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    
    print(f"\nComparison chart successfully generated and saved to: {out_path}")

if __name__ == "__main__":
    main()
