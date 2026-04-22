"""
reproduce_results_offline.py
============================
Offline evaluation script utilizing the Unified Sequence Engine (seq_model.joblib).
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

MODELS_DIR = os.path.join(BASE_DIR, "models")
SEQ_MODEL_PATH = os.path.join(MODELS_DIR, "seq_model.joblib")
TEST_SPLIT_PATH = os.path.join(BASE_DIR, "samples_test_split.json")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
MANIFEST_PATH = os.path.join(BASE_DIR, "manifest.json")

CLASSES = ["telegram", "slack", "discord"]

def _evaluate_single_chain(cid, fnames, gt_chains, samples_dir, seq_model):
    """Worker function for parallel chain evaluation."""
    from core.forensic_features import ForensicFeatureExtractor
    extractor = ForensicFeatureExtractor()
    
    step1_fname = next((f for f in fnames if ".step1." in f), None)
    step2_fname = next((f for f in fnames if ".step2." in f), None)
    step3_fname = next((f for f in fnames if ".step3." in f), None)
    if not (step1_fname and step2_fname and step3_fname): 
        return None
        
    true_seq = gt_chains.get(cid)
    if not true_seq: 
        return None

    vecs = []
    for fname in [step1_fname, step2_fname, step3_fname]:
        fpath = os.path.join(samples_dir, fname)
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
        "cid": cid,
        "true_seq": true_seq,
        "pred_seq": pred_seq,
        "match": true_seq == pred_seq
    }

def _fail(title: str, msg: str):
    print(f"\n[!] {title}\n    {msg}\n")
    sys.exit(1)

def main():
    if not os.path.isfile(SEQ_MODEL_PATH):
        _fail("Model missing", "Run 'python scripts/export_models.py' first.")

    try:
        seq_model = joblib.load(SEQ_MODEL_PATH)
    except Exception as e:
        _fail("Model Load Failed", f"Could not load {SEQ_MODEL_PATH}: {e}")

    # Load ground truth chains
    try:
        with open(MANIFEST_PATH, "r") as mf:
            manifest = json.load(mf)
    except FileNotFoundError:
        _fail("Missing manifest", f"Could not find {MANIFEST_PATH}")

    gt_chains = {entry["chain_id"]: entry.get("sequence", []) for entry in manifest if entry.get("chain_id") and len(entry.get("sequence", [])) == 3}

    # Load test split
    try:
        with open(TEST_SPLIT_PATH, "r") as f:
            test_files = json.load(f)
    except FileNotFoundError:
        _fail("Missing Test Split", f"Could not find {TEST_SPLIT_PATH}")

    # Group test files by chain
    test_chains = {}
    for fname in test_files:
        try:
            cid = fname.split(".chain_")[1].split(".step")[0]
            if cid not in test_chains:
                test_chains[cid] = []
            test_chains[cid].append(fname)
        except Exception:
            pass

    extractor = ForensicFeatureExtractor()

    print("\nStarting Parallel Offline Sequence Evaluation (Unified v2)...")
    
    results = Parallel(n_jobs=-1)(
        delayed(_evaluate_single_chain)(cid, fnames, gt_chains, SAMPLES_DIR, seq_model)
        for cid, fnames in test_chains.items()
    )
    
    processed_results = [r for r in results if r is not None]
    
    total_chains = 0
    exact_matches = 0
    step_correct = {0: 0, 1: 0, 2: 0}

    print("\n  [Chain ID]         [True Sequence]                [Predicted Sequence]")
    print("  " + "-"*75)

    for r in processed_results:
        true_str = ">".join(r["true_seq"])
        pred_str = ">".join(r["pred_seq"])
        match_flag = "OK" if r["match"] else "--"
        print(f"  {r['cid'][:18]:<18} {true_str:<30} {pred_str:<25} [{match_flag}]")

        total_chains += 1
        if r["match"]:
            exact_matches += 1
            
        for i in range(3):
            if r["true_seq"][i] == r["pred_seq"][i]:
                step_correct[i] += 1

    print("\n" + "="*60)
    if total_chains > 0:
        print(f"  Total Chains Evaluated: {total_chains}")
        print(f"  Step 1 Accuracy     : {step_correct[0]/total_chains * 100:.1f}%")
        print(f"  Step 2 Accuracy     : {step_correct[1]/total_chains * 100:.1f}%")
        print(f"  Step 3 Accuracy     : {step_correct[2]/total_chains * 100:.1f}%")
        print(f"  Exact Sequence Match: {exact_matches/total_chains * 100:.1f}%")
    else:
        print("  No chains evaluated.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
