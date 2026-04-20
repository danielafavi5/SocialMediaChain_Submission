"""
export_models.py
================
Trains the Unified Forensic Sequence Engine (V2) via MultiOutputClassifier.
Replaces the legacy C1/C2 pipeline.
"""

import os
import sys
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib

# Dynamic relative paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.forensic_features import ForensicFeatureExtractor

SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
MANIFEST_PATH = os.path.join(BASE_DIR, "manifest.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SEQ_MODEL_PATH = os.path.join(MODELS_DIR, "seq_model.joblib")
TEST_SPLIT_PATH = os.path.join(BASE_DIR, "samples_test_split.json")
CACHE_FILE = os.path.join(BASE_DIR, "unified_sequence_cache.npz")
PRUNED_IDX_PATH = os.path.join(MODELS_DIR, "pruned_indices.npy")

# Adversarial pruning: 15 volatile features per step, applied to all 3 steps in the 816-dim chain vector
_DRIFT_BASE = [225, 224, 138, 139, 145, 150, 152, 142, 143, 156, 126, 127, 128, 129, 130]
KS_TOP_DRIFT_INDICES = (
    _DRIFT_BASE +
    [x + 272 for x in _DRIFT_BASE] +
    [x + 544 for x in _DRIFT_BASE]
)

CLASS_MAP = {"telegram": 0, "slack": 1, "discord": 2}

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("Building Seq2Seq Model from samples/ (272 dims)...")
    ext = ForensicFeatureExtractor()

    with open(MANIFEST_PATH, "r") as mf:
        manifest = json.load(mf)
        
    gt_chains = {}
    for e in manifest:
        cid = e.get("chain_id")
        seq = e.get("sequence", [])
        if cid and len(seq) == 3:
            gt_chains[cid] = seq

    jpgs = sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(".jpg"))
    chains = {}
    for f in jpgs:
        try:
            cid = f.split(".chain_")[1].split(".step")[0]
            if cid not in chains: chains[cid] = []
            chains[cid].append(f)
        except Exception:
            pass

    chain_ids = sorted(list(gt_chains.keys()))
    train_cids, test_cids = train_test_split(chain_ids, test_size=0.2, random_state=42)

    # Save test files for offline evaluation
    test_files = []
    for cid in test_cids:
        test_files.extend(chains.get(cid, []))
    with open(TEST_SPLIT_PATH, "w") as f:
        json.dump(test_files, f, indent=2)
    print(f"  Saved test split to {TEST_SPLIT_PATH}")

    X_list, y_list = [], []
    
    print("  Extracting Train Features (full chain: step1+step2+step3 concatenated)...")
    # Extract all 3 steps and concatenate into a 816-dim chain signature
    for i, cid in enumerate(train_cids, 1):
        chain_files = chains.get(cid, [])
        step1_file = next((f for f in chain_files if ".step1." in f), None)
        step2_file = next((f for f in chain_files if ".step2." in f), None)
        step3_file = next((f for f in chain_files if ".step3." in f), None)
        if not (step1_file and step2_file and step3_file): continue
        
        seq = gt_chains[cid]
        try:
            lbls = [CLASS_MAP[seq[0]], CLASS_MAP[seq[1]], CLASS_MAP[seq[2]]]
        except KeyError:
            continue
            
        vecs = []
        valid = True
        for fname in [step1_file, step2_file, step3_file]:
            fvec = ext.extract(os.path.join(SAMPLES_DIR, fname))
            if not np.count_nonzero(fvec):
                valid = False
                break
            vecs.append(fvec)
        
        if not valid:
            continue
            
        chain_sig = np.concatenate(vecs)  # 816-dim (272*3)
        X_list.append(chain_sig)
        y_list.append(lbls)

    X = np.vstack(X_list)
    y = np.array(y_list)
    
    feat_dim = X.shape[1]
    print(f"  Samples used: {X.shape[0]} | Feature dims (raw): {feat_dim}")

    # Adversarial Pruning implementation guardrail
    valid_indices = [i for i in range(feat_dim) if i not in KS_TOP_DRIFT_INDICES]
    X_pruned = X[:, valid_indices]
    np.save(PRUNED_IDX_PATH, np.array(valid_indices))
    print(f"  Pruned adversarial elements to: {X_pruned.shape[1]} dimensions")

    # Cache Naming Convention Guardrail
    np.savez_compressed(CACHE_FILE, X_train=X_pruned, y_seq_labels=y)
    print(f"  Saved unified cache to {CACHE_FILE}")

    # Train Seq2Seq Model
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_pruned, y, test_size=0.2, random_state=42
    )

    base_rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=1)
    seq_model = MultiOutputClassifier(base_rf)
    
    seq_model.fit(X_tr, y_tr)
    y_pred_val = seq_model.predict(X_val)
    
    # Evaluate sequence-level macro F1 (approximation per step)
    f1_step1 = f1_score(y_val[:, 0], y_pred_val[:, 0], average="macro")
    f1_step2 = f1_score(y_val[:, 1], y_pred_val[:, 1], average="macro")
    f1_step3 = f1_score(y_val[:, 2], y_pred_val[:, 2], average="macro")
    print(f"  Validation macro-F1: [Step1: {f1_step1:.3f}, Step2: {f1_step2:.3f}, Step3: {f1_step3:.3f}]")
    
    # Exact full-sequence match accuracy
    seq_acc = np.mean(np.all(y_pred_val == y_val, axis=1))
    print(f"  Validation Exact Sequence Match: {seq_acc*100:.1f}%")

    seq_model.fit(X_pruned, y)
    joblib.dump(seq_model, SEQ_MODEL_PATH)
    print(f"  Saved end-to-end Sequence Model to {SEQ_MODEL_PATH}")

if __name__ == "__main__":
    main()
