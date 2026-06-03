import os
import sys
import subprocess
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

    # Parse sample predictions using a precise regular expression
    # Row pattern:   01804966_177151723 telegram>slack>discord         telegram>discord>slack    [--]
    pattern = re.compile(r"^\s*([a-zA-Z0-9_]+)\s+([a-z_>]+)\s+([a-z_>]+)\s+\[(OK|--)\]")
    
    samples = []
    
    for line in output.splitlines():
        match = pattern.match(line)
        if match:
            cid = match.group(1)
            true_seq = match.group(2).replace('>', ' \u2192 ') # Render with Unicode arrows: →
            pred_seq = match.group(3).replace('>', ' \u2192 ')
            is_match = "True" if match.group(4) == "OK" else "False"
            
            samples.append({
                "id": cid,
                "true": true_seq,
                "pred": pred_seq,
                "match": is_match
            })
            
            # Select only a small subset (first 10 samples)
            if len(samples) == 10:
                break
                
    if not samples:
        print("Error: Could not parse any sequences from the evaluation output.")
        return
        
    print(f"Parsed {len(samples)} sample records for the visualization table.")
    
    # Setup canvas (16:9 aspect ratio)
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='#FCFCFC')
    ax.set_facecolor('#FCFCFC')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    # 1. Draw Title
    plt.text(8.0, 8.2, "Offline Sequence Prediction Examples", 
             fontsize=18, fontweight='bold', ha='center', color='#2C3E50')
    plt.text(8.0, 7.8, "First 10 Chain Predictions from the Leak-Free Holdout Set", 
             fontsize=12, fontstyle='italic', ha='center', color='#7F8C8D')
    
    # 2. Draw Table Headers and Grid
    # Column coordinates: (col_x_start, col_width)
    cols = [
        {"name": "Sample ID", "x": 1.0, "w": 3.0, "align": "center"},
        {"name": "True Sequence", "x": 4.1, "w": 4.8, "align": "left"},
        {"name": "Predicted Sequence", "x": 9.0, "w": 4.8, "align": "left"},
        {"name": "Match", "x": 13.9, "w": 1.1, "align": "center"}
    ]
    
    # Table vertical coordinates
    header_y = 6.9
    header_h = 0.5
    row_h = 0.48
    row_y_start = 6.4
    
    # Draw Header background card
    header_bg = patches.Rectangle((1.0, header_y), 14.0, header_h, 
                                  facecolor='#2C3E50', edgecolor='none')
    ax.add_patch(header_bg)
    
    # Draw Header text
    for col in cols:
        ha = col["align"]
        if ha == "center":
            text_x = col["x"] + col["w"] / 2.0
        else:
            text_x = col["x"] + 0.2
            
        plt.text(text_x, header_y + 0.15, col["name"], 
                 fontsize=11.5, fontweight='bold', color='#FFFFFF', ha=ha, va='bottom')
                 
    # Draw Row details
    for r, sample in enumerate(samples):
        y_pos = row_y_start - (r * row_h)
        
        # Alternating row background colors
        bg_color = '#F9FBFD' if r % 2 == 0 else '#FFFFFF'
        row_bg = patches.Rectangle((1.0, y_pos), 14.0, row_h, 
                                   facecolor=bg_color, edgecolor='#E5E9F0', linewidth=0.5)
        ax.add_patch(row_bg)
        
        # Draw cells
        for c, col in enumerate(cols):
            ha = col["align"]
            if ha == "center":
                text_x = col["x"] + col["w"] / 2.0
            else:
                text_x = col["x"] + 0.2
                
            # Column-specific text and coloring
            if col["name"] == "Sample ID":
                text_str = sample["id"]
                text_color = '#34495E'
                is_bold = False
            elif col["name"] == "True Sequence":
                text_str = sample["true"]
                text_color = '#2C3E50'
                is_bold = False
            elif col["name"] == "Predicted Sequence":
                text_str = sample["pred"]
                text_color = '#2C3E50'
                is_bold = False
            elif col["name"] == "Match":
                text_str = sample["match"]
                text_color = '#2E7D32' if text_str == "True" else '#C62828'
                is_bold = True
                
            plt.text(
                text_x, 
                y_pos + 0.16, 
                text_str, 
                fontsize=10.5, 
                fontweight='bold' if is_bold else 'normal', 
                color=text_color, 
                ha=ha, 
                va='bottom'
            )
            
    # Add a thin outer table border
    outer_border = patches.Rectangle((1.0, row_y_start - 9 * row_h), 14.0, header_h + 10 * row_h, 
                                     facecolor='none', edgecolor='#BDC3C7', linewidth=1.0)
    ax.add_patch(outer_border)
    
    # 3. Save Figure
    plt.tight_layout()
    
    os.makedirs('assets', exist_ok=True)
    out_path = os.path.join('assets', 'example_predictions_table.png')
    plt.savefig(out_path, dpi=300, facecolor='#FCFCFC', bbox_inches='tight')
    
    print(f"\nExample predictions table successfully saved to: {out_path}")

if __name__ == "__main__":
    main()
