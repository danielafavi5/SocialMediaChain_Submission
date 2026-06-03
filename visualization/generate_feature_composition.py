import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

def main():
    print("Generating 272-Dimensional Feature Vector Composition Diagram...")
    
    # Define exact feature families and their dimensional contributions based on forensic_features.py
    feature_families = [
        {
            "name": "DCT AC Histogram Bins",
            "dims": 21,
            "range": "[0:21]",
            "desc": "AC coefficient magnitude histogram bins capturing compression scale"
        },
        {
            "name": "DCT Coefficient Energy Map",
            "dims": 21,
            "range": "[21:42]",
            "desc": "AC energy normalized by DC energy representing frequency energy distribution"
        },
        {
            "name": "Intra-block Markov Transitions",
            "dims": 81,
            "range": "[42:123]",
            "desc": "Quantized transition probability matrix (T=4) capturing block boundary signatures"
        },
        {
            "name": "Luminance Q-table",
            "dims": 64,
            "range": "[123:187]",
            "desc": "Standard 8x8 Luma quantization coefficients from DQT segment"
        },
        {
            "name": "Luma Q-table Stats",
            "dims": 4,
            "range": "[187:191]",
            "desc": "Basic statistical features of Luma table: [mean, std, min, max]"
        },
        {
            "name": "Chrominance Q-table",
            "dims": 40,
            "range": "[191:231]",
            "desc": "First 40 Chroma quantization coefficients representing color subsampling"
        },
        {
            "name": "Chroma Q-table Stats",
            "dims": 4,
            "range": "[231:235]",
            "desc": "Basic statistical features of Chroma table: [mean, std, min, max]"
        },
        {
            "name": "Metadata Flags",
            "dims": 6,
            "range": "[235:241]",
            "desc": "EXIF presence, datetime tags, software hashes, and JPEG marker flags"
        },
        {
            "name": "Structural Stats",
            "dims": 6,
            "range": "[241:247]",
            "desc": "Image height/width, compression ratio, Y-channel mean/std, and blocking effect"
        },
        {
            "name": "Ghost Peaks",
            "dims": 5,
            "range": "[247:252]",
            "desc": "Double-compression AC1-AC5 empty bins ratio capturing re-save artifacts"
        },
        {
            "name": "Q-Table Backtracking",
            "dims": 6,
            "range": "[252:258]",
            "desc": "Normalized L1 distances to 6 standard quantization tables (FB, Flickr, TG, etc.)"
        },
        {
            "name": "Benford's Law Analysis",
            "dims": 9,
            "range": "[258:267]",
            "desc": "First-digit probability distribution of AC coefficients"
        },
        {
            "name": "Container Byte Analysis",
            "dims": 5,
            "range": "[267:272]",
            "desc": "Presence and order of APP0, APP1, DHT, and DQT byte markers"
        }
    ]
    
    # Distinct 13-color palette
    colors = [
        '#4A90E2', # DCT AC Hist - blue
        '#5C258D', # DCT Energy - purple
        '#4389A2', # Intra-block Markov - teal
        '#08AEEA', # Luma Q-table - cyan
        '#2AF598', # Luma stats - light green
        '#667EEA', # Chroma Q-table - indigo
        '#F5576C', # Chroma stats - rose
        '#F093FB', # Meta - lavender pink
        '#F6D365', # Structural - amber gold
        '#4FACFE', # Ghost peaks - sky blue
        '#00C6FF', # Q-table Backtracking - electric blue
        '#7F00FF', # Benford - violet
        '#FF007F'  # Container - magenta pink
    ]
    
    # Setup canvas (16:9 aspect ratio)
    fig, ax = plt.subplots(figsize=(16, 9.5), facecolor='#FCFCFC')
    ax.set_facecolor('#FCFCFC')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    # 1. Draw Title
    plt.text(8.0, 8.4, "Unified Sequence Engine: Feature Vector Composition Map", 
             fontsize=18, fontweight='bold', ha='center', color='#2C3E50')
    plt.text(8.0, 8.1, "Proportional Layout of the 272-Dimensional Forensic Feature Vector", 
             fontsize=12, fontstyle='italic', ha='center', color='#7F8C8D')
    
    # 2. Draw Feature Vector Tape (y-coordinate from 7.0 to 7.5, spanning x from 1.0 to 15.0)
    tape_y = 7.1
    tape_h = 0.5
    tape_start_x = 1.0
    tape_width_x = 14.0
    
    accum_dims = 0
    
    for i, fam in enumerate(feature_families):
        dims = fam["dims"]
        start_idx = accum_dims
        end_idx = accum_dims + dims
        
        # Calculate canvas positions
        x_start = tape_start_x + (start_idx / 272.0) * tape_width_x
        x_end = tape_start_x + (end_idx / 272.0) * tape_width_x
        width = x_end - x_start
        
        # Draw tape segment rectangle
        rect = patches.Rectangle((x_start, tape_y), width, tape_h, 
                                 facecolor=colors[i], edgecolor='#FFFFFF', linewidth=1.0)
        ax.add_patch(rect)
        
        # Add labels to major/wider segments inside the tape
        if dims >= 20:
            plt.text(x_start + width/2.0, tape_y + 0.18, f"{dims}", 
                     fontsize=9, color='#FFFFFF', fontweight='bold', ha='center')
            
        # Draw major boundaries below tape
        if start_idx in [0, 42, 123, 187, 231, 272] or i == len(feature_families)-1:
            plt.text(x_start, tape_y - 0.25, f"{start_idx}", 
                     fontsize=9, color='#7F8C8D', fontweight='bold', ha='center')
            ax.plot([x_start, x_start], [tape_y, tape_y - 0.1], color='#BDC3C7', linewidth=1.0)
            
        accum_dims += dims
        
    # Draw final boundary tick
    plt.text(tape_start_x + tape_width_x, tape_y - 0.25, "272", 
             fontsize=9, color='#7F8C8D', fontweight='bold', ha='center')
    ax.plot([tape_start_x + tape_width_x, tape_start_x + tape_width_x], [tape_y, tape_y - 0.1], 
            color='#BDC3C7', linewidth=1.0)
            
    # Labels for Tape
    plt.text(tape_start_x - 0.1, tape_y + tape_h/2.0, "Vector Tape", 
             fontsize=10, fontweight='bold', color='#34495E', ha='right', va='center')
             
    # 3. Draw Cards in 2 Columns
    # Col 1: Card 1 to 7
    # Col 2: Card 8 to 13
    col1_x = 1.0
    col2_x = 8.3
    card_w = 6.7
    card_h = 0.72
    
    col1_y_start = 5.8
    col2_y_start = 5.8
    y_step = 0.82
    
    for i, fam in enumerate(feature_families):
        # Determine column and card index
        if i < 7:
            x_pos = col1_x
            y_pos = col1_y_start - (i * y_step)
        else:
            x_pos = col2_x
            y_pos = col2_y_start - ((i - 7) * y_step)
            
        # Draw Card container (White background, light gray border)
        card_rect = patches.Rectangle((x_pos, y_pos), card_w, card_h, 
                                      facecolor='#FFFFFF', edgecolor='#E5E9F0', 
                                      linewidth=1.0)
        ax.add_patch(card_rect)
        
        # Draw solid left color accent stripe (stripe width = 0.12)
        stripe_w = 0.12
        stripe_rect = patches.Rectangle((x_pos, y_pos), stripe_w, card_h, 
                                        facecolor=colors[i], edgecolor='none')
        ax.add_patch(stripe_rect)
        
        # Add Text within the card
        # Title of the family
        plt.text(x_pos + 0.25, y_pos + 0.46, fam["name"], 
                 fontsize=10.5, fontweight='bold', color='#2C3E50', va='bottom')
                 
        # Dimension and indices details
        plt.text(x_pos + 0.25, y_pos + 0.30, f"{fam['dims']} Dimensions | Indices {fam['range']}", 
                 fontsize=8.5, fontweight='bold', color=colors[i], va='bottom')
                 
        # Description
        plt.text(x_pos + 0.25, y_pos + 0.08, fam["desc"], 
                 fontsize=8, color='#7F8C8D', va='bottom')
                 
    # Polish layouts and save
    plt.tight_layout()
    
    os.makedirs('assets', exist_ok=True)
    out_path = os.path.join('assets', 'feature_composition.png')
    plt.savefig(out_path, dpi=300, facecolor='#FCFCFC', bbox_inches='tight')
    
    print(f"\nFeature composition map successfully saved to: {out_path}")

if __name__ == "__main__":
    main()
