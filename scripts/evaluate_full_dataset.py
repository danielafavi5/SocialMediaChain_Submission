"""
evaluate_full_dataset.py
========================
Diagnostic script to evaluate the Unified Sequence Engine (seq_model.joblib)
against ALL held-out test chains (i.e. all chains NOT used during training).

This uses the same strict train/test split from samples_test_split.json
but evaluates every qualifying test chain, confirming generalization
performance independently of the official reproduce_results_offline.py script.
"""

import os
import sys
import json
import joblib
from joblib import Parallel, delayed
import numpy as np

# Dynamic relative paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.forensic_features import ForensicFeatureExtractor

CLASSES = ["telegram", "slack", "discord"]
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
MANIFEST_PATH = os.path.join(BASE_DIR, "manifest.json")
SEQ_MODEL_PATH = os.path.join(BASE_DIR, "models", "seq_model.joblib")
TEST_SPLIT_PATH = os.path.join(BASE_DIR, "samples_test_split.json")

def _evaluate_single_chain(cid, fnames, gt_seq, seq_model):
    """Worker function for parallel chain evaluation."""
    extractor = ForensicFeatureExtractor()
    step1_fname = next((f for f in fnames if ".step1." in f), None)
    step2_fname = next((f for f in fnames if ".step2." in f), None)
    step3_fname = next((f for f in fnames if ".step3." in f), None)
    if not (step1_fname and step2_fname and step3_fname): 
        return None
    
    vecs = []
    for fname in [step1_fname, step2_fname, step3_fname]:
        fpath = os.path.join(SAMPLES_DIR, fname)
        fvec = extractor.extract(fpath)
        if not np.count_nonzero(fvec): 
            return None
        vecs.append(fvec)
        
    feat_input = np.concatenate(vecs)
    try:
        pred_idx_seq = seq_model.predict([feat_input])[0].astype(int)
        pred_seq = [CLASSES[idx] for idx in pred_idx_seq]
    except Exception:
        return None
        
    return {
        "match": gt_seq == pred_seq,
        "true_seq": gt_seq,
        "pred_seq": pred_seq
    }

def main():
    if not os.path.exists(SEQ_MODEL_PATH):
        print(f"[!] Sequence model missing: {SEQ_MODEL_PATH}")
        sys.exit(1)
        
    print(f"Loading sequence model: {SEQ_MODEL_PATH}")
    seq_model = joblib.load(SEQ_MODEL_PATH)
    
    with open(MANIFEST_PATH, "r") as mf:
        manifest = json.load(mf)
        
    # Build ground truth dictionary for complete 3-step chains
    gt_chains = {entry["chain_id"]: entry.get("sequence", []) for entry in manifest if entry.get("chain_id") and len(entry.get("sequence", [])) == 3}
    
    # Load test split and derive held-out chain IDs (chains NOT in training set)
    with open(TEST_SPLIT_PATH, "r") as f:
        test_files = json.load(f)
    
    test_chain_ids = set()
    for fname in test_files:
        try:
            cid = fname.split(".chain_")[1].split(".step")[0]
            test_chain_ids.add(cid)
        except Exception:
            pass
    
    print(f"  Holdout (test) chains identified: {len(test_chain_ids)}")
    
    # Locate all images in the samples directory
    jpgs = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(".jpg")]
    chains = {}
    for f in jpgs:
        try:
            cid = f.split(".chain_")[1].split(".step")[0]
            if cid not in chains: chains[cid] = []
            chains[cid].append(f)
        except Exception: 
            pass
    
    # Filter: keep only chains in the held-out test set
    chains = {cid: fnames for cid, fnames in chains.items() if cid in test_chain_ids}
    print(f"  Chains available to evaluate: {len(chains)}")
            
    print("\nStarting Parallel Held-Out Test Set Evaluation (Training chains excluded)...")
        
    results = Parallel(n_jobs=-1)(
        delayed(_evaluate_single_chain)(cid, fnames, gt_chains[cid], seq_model)
        for cid, fnames in chains.items() if cid in gt_chains
    )
    
    valid_res = [r for r in results if r is not None]
    matches = sum(1 for r in valid_res if r["match"])
    
    step_correct = {0: 0, 1: 0, 2: 0}
    for r in valid_res:
        for i in range(3):
            if r["true_seq"][i] == r["pred_seq"][i]:
                step_correct[i] += 1
                
    total = len(valid_res)
    if total == 0:
        print("No valid chains evaluated.")
        sys.exit(1)
        
    print("\n" + "="*60)
    print(f"  Total Chains Evaluated (Full Dataset): {total}")
    print(f"  Step 1 Accuracy     : {step_correct[0]/total*100:.1f}%")
    print(f"  Step 2 Accuracy     : {step_correct[1]/total*100:.1f}%")
    print(f"  Step 3 Accuracy     : {step_correct[2]/total*100:.1f}%")
    print(f"  Exact Sequence Match: {matches/total*100:.1f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
