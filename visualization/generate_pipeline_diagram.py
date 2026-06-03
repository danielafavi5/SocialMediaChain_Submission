import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    print("Generating Pipeline Flowchart Diagram...")
    
    # Setup canvas (16:9 aspect ratio)
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='#FCFCFC')
    ax.set_facecolor('#FCFCFC')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    # Draw Title
    plt.text(8.0, 7.8, "Unified Sequence Engine (V2): Pipeline Architecture", 
             fontsize=18, fontweight='bold', ha='center', color='#2C3E50')
    plt.text(8.0, 7.4, "End-to-end workflow from raw image to predicted sequence", 
             fontsize=12, fontstyle='italic', ha='center', color='#7F8C8D')
             
    # Define the 5 stages
    stages = [
        {
            "title": "Data Ingestion",
            "desc": "JPEG Image Input\n(Any Resolution)",
            "color": "#95A5A6" # Grey
        },
        {
            "title": "Feature Extraction",
            "desc": "Forensic Analyzer\n(272-D Vector)",
            "color": "#3498DB" # Blue
        },
        {
            "title": "Sequential Model",
            "desc": "Random Forest\nClassifier Chain",
            "color": "#2ECC71" # Green
        },
        {
            "title": "BKS Traceback",
            "desc": "Knowledge Fusion\n(Graph Search)",
            "color": "#F39C12" # Orange
        },
        {
            "title": "Output Chain",
            "desc": "Predicted Sequence\n(e.g. TG \u2192 SL \u2192 DC)",
            "color": "#9B59B6" # Purple
        }
    ]
    
    # Layout parameters
    box_w = 2.4
    box_h = 1.4
    start_x = 0.5
    spacing = 0.7
    y_pos = 4.0
    
    for i, stage in enumerate(stages):
        x_pos = start_x + i * (box_w + spacing)
        
        # Draw Box (with subtle shadow)
        shadow = patches.FancyBboxPatch((x_pos + 0.05, y_pos - 0.05), box_w, box_h,
                                        boxstyle="round,pad=0.1", 
                                        facecolor='#E0E0E0', edgecolor='none')
        ax.add_patch(shadow)
        
        box = patches.FancyBboxPatch((x_pos, y_pos), box_w, box_h,
                                     boxstyle="round,pad=0.1", 
                                     facecolor=stage["color"], edgecolor='#FFFFFF', linewidth=2)
        ax.add_patch(box)
        
        # Draw Text inside Box
        plt.text(x_pos + box_w/2.0, y_pos + box_h/2.0 + 0.2, stage["title"], 
                 fontsize=12, fontweight='bold', color='#FFFFFF', ha='center', va='center')
        plt.text(x_pos + box_w/2.0, y_pos + box_h/2.0 - 0.25, stage["desc"], 
                 fontsize=10, color='#FFFFFF', ha='center', va='center')
                 
        # Draw Arrow to the next box
        if i < len(stages) - 1:
            arrow_start_x = x_pos + box_w + 0.1
            arrow_end_x = arrow_start_x + spacing - 0.2
            arrow_y = y_pos + box_h/2.0
            
            arrow = patches.FancyArrowPatch((arrow_start_x, arrow_y), (arrow_end_x, arrow_y),
                                            mutation_scale=20, color='#BDC3C7', linewidth=2.5)
            ax.add_patch(arrow)
            
    # Save the diagram
    plt.tight_layout()
    os.makedirs('assets', exist_ok=True)
    out_path = os.path.join('assets', 'pipeline_flowchart.png')
    plt.savefig(out_path, dpi=300, facecolor='#FCFCFC', bbox_inches='tight')
    
    print(f"Pipeline Flowchart successfully saved to: {out_path}")

if __name__ == "__main__":
    main()
