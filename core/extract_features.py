"""
extract_features.py
======================
Phase 7: Augmented Chain Training - Feature Extraction

Extracts the 258-dimensional features exclusively from the intermediate chain steps
(Step 1 and Step 2) generated in 2026. This data will be used to train the
History-Aware Residual Random Forest, allowing it to mathematically understand
what "Telegram->Discord" structural boundaries actually look like, breaking the Loop Collapse.
"""

import os
import glob
import time
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

from forensic_features import ForensicFeatureExtractor, FEATURE_DIM

RESULTS_2026 = Path("results_2026/results_2026")
CACHE_FILE = "phase7_features_cache.npz"

import tempfile
from PIL import Image

def extract_single(img_path):
    ext = ForensicFeatureExtractor()
    fname = os.path.basename(img_path)
    
    # 1. Ghost Simulation: Simulate subsequent platform compression
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_name = tmp.name
        
    try:
        img = Image.open(img_path)
        if img.mode not in ("RGB", "L", "YCbCr"): img = img.convert("RGB")
        img.save(tmp_name, format="JPEG", quality=75)
        
        vec = ext.extract(tmp_name)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
            
    # 2. Extract chain_id for the leak-free test split
    chain_id = "unknown"
    try:
        chain_id = fname.split(".chain_")[1].split(".step")[0]
    except Exception:
        pass
        
    return fname, chain_id, vec

def main():
    print(f"Phase 7 Augmented Extraction | Feature Dim = {FEATURE_DIM}")
    
    # We need Step 1 AND Step 2 images from the chains.
    # To keep classes balanced with Phase 5, we will extract intermediate steps
    # and label them based on their *Current* platform in that step.
    
    files_s1 = glob.glob(str(RESULTS_2026 / "*.step1.*.jpg"))
    files_s2 = glob.glob(str(RESULTS_2026 / "*.step2.*.jpg"))
    
    # Sample up to say 300 of each
    np.random.seed(42)
    files_to_process = list(np.random.choice(files_s1, min(len(files_s1), 300), replace=False)) + \
                       list(np.random.choice(files_s2, min(len(files_s2), 300), replace=False))
                       
    print(f"Extracting {len(files_to_process)} Augmented Chain Intermediate Images...")
    
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as exe:
        futures = {exe.submit(extract_single, f): f for f in files_to_process}
        for fut in as_completed(futures):
            fname, cid, vec = fut.result()
            if np.count_nonzero(vec) > 0:
                results.append((fname, cid, vec))
                
    print(f"  -> Done in {time.time()-t0:.1f}s. Valid: {len(results)}")
    
    # Parse labels from filename: D01_I_nat_0001.chain_xxxx.step2.telegram.jpg
    X = []
    labels = []
    cids = []
    
    for fname, cid, vec in results:
        plat = fname.split('.')[3] # platform is the second-to-last dot-separated token (e.g. '...step2.telegram.jpg')
        X.append(vec)
        labels.append(plat)
        cids.append(cid)
        
    X = np.array(X, dtype=np.float32)
    labels = np.array(labels)
    cids = np.array(cids)
    
    np.savez_compressed(
        CACHE_FILE,
        X_aug=X,
        labels_aug=labels,
        cids_aug=cids
    )
    print(f"Saved Augmented Cache -> {CACHE_FILE} (Shape: {X.shape})")

if __name__ == "__main__":
    main()
