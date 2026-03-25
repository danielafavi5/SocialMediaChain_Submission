import os
import glob
import json

def main():
    samples_dir = "samples"
    manifest_path = "manifest.json"
    
    jpgs = sorted(glob.glob(os.path.join(samples_dir, "*.jpg")))
    
    chains = {}
    for f in jpgs:
        fname = os.path.basename(f)
        # Parse: D01_I_nat_0001.chain_502c0eb0_1771466528.step1.telegram.jpg
        try:
            parts = fname.split('.chain_')
            orig_image = parts[0] + ".jpg"
            rest = parts[1].split('.')
            chain_id = rest[0]
            step = int(rest[1].replace('step', ''))
            platform = rest[2]
            
            if chain_id not in chains:
                chains[chain_id] = []
            chains[chain_id].append({
                "chain_id": chain_id,
                "step": step,
                "platform": platform,
                "orig_image": orig_image,
                "served_filename": fname
            })
        except Exception as e:
            print(f"Failed to parse {fname}: {e}")
            
    manifest = []
    for cid, file_entries in chains.items():
        # Sort by step
        file_entries.sort(key=lambda x: x["step"])
        sequence = [e["platform"] for e in file_entries]
        
        for e in file_entries:
            e["sequence"] = sequence
            manifest.append(e)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Generated manifest.json with {len(manifest)} entries.")

if __name__ == "__main__":
    main()
