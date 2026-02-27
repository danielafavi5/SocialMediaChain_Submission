"""
phase7_data_extract.py
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

RESULTS_2026 = Path("results_2026")
CACHE_FILE = "phase7_features_cache.npz"

def extract_single(img_path):
    ext = ForensicFeatureExtractor()
    return os.path.basename(img_path), ext.extract(img_path)

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
            fname, vec = fut.result()
            if np.count_nonzero(vec) > 0:
                results.append((fname, vec))
                
    print(f"  -> Done in {time.time()-t0:.1f}s. Valid: {len(results)}")
    
    # Parse labels from filename: D01_I_nat_0001.chain_xxxx.step2.telegram.jpg
    X = []
    labels = []
    
    for fname, vec in results:
        plat = fname.split('.')[3] # platform is the 4th token
        X.append(vec)
        labels.append(plat)
        
    X = np.array(X, dtype=np.float32)
    labels = np.array(labels)
    
    np.savez_compressed(
        CACHE_FILE,
        X_aug=X,
        labels_aug=labels
    )
    print(f"Saved Augmented Cache -> {CACHE_FILE} (Shape: {X.shape})")

if __name__ == "__main__":
    main()
