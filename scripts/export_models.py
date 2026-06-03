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
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import f1_score
import joblib
from joblib import Parallel, delayed

# Dynamic relative paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.forensic_features import ForensicFeatureExtractor

SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
MANIFEST_PATH = os.path.join(BASE_DIR, "manifest.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SEQ_MODEL_PATH = os.path.join(MODELS_DIR, "seq_model.joblib")
SURF_MODEL_PATH = os.path.join(MODELS_DIR, "surface_model.joblib")
TEST_SPLIT_PATH = os.path.join(BASE_DIR, "samples_test_split.json")
CACHE_FILE = os.path.join(BASE_DIR, "unified_sequence_cache.npz")
PRUNED_IDX_PATH = os.path.join(MODELS_DIR, "pruned_indices.npy")

CLASS_MAP = {"telegram": 0, "slack": 1, "discord": 2}

def _process_single_chain(cid, chain_files, gt_chains, samples_dir):
    """Worker function for parallel feature extraction."""
    from core.forensic_features import ForensicFeatureExtractor
    ext = ForensicFeatureExtractor()
    
    step1_file = next((f for f in chain_files if ".step1." in f), None)
    step2_file = next((f for f in chain_files if ".step2." in f), None)
    step3_file = next((f for f in chain_files if ".step3." in f), None)
    
    if not (step1_file and step2_file and step3_file): 
        return None
        
    seq = gt_chains[cid]
    try:
        lbls = [CLASS_MAP[seq[0]], CLASS_MAP[seq[1]], CLASS_MAP[seq[2]]]
    except KeyError:
        return None
            
    vecs = []
    for fname in [step1_file, step2_file, step3_file]:
        fvec = ext.extract(os.path.join(samples_dir, fname))
        if not np.count_nonzero(fvec):
            return None
        vecs.append(fvec)
        
    source_id = step1_file.split(".chain_")[0] if step1_file else "unknown"
    return np.concatenate(vecs), lbls, source_id

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
    
    source_ids_all = []
    for cid in chain_ids:
        chain_files = chains.get(cid, [])
        if chain_files:
            step1_file = next((f for f in chain_files if ".step1." in f), chain_files[0])
            source_id = step1_file.split(".chain_")[0]
            source_ids_all.append(source_id)
        else:
            source_ids_all.append("unknown")
            
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(chain_ids, groups=source_ids_all))
    
    train_cids = [chain_ids[i] for i in train_idx]
    test_cids = [chain_ids[i] for i in test_idx]

    # Save test files for offline evaluation
    test_files = []
    for cid in test_cids:
        test_files.extend(chains.get(cid, []))
    with open(TEST_SPLIT_PATH, "w") as f:
        json.dump(test_files, f, indent=2)
    print(f"  Saved test split to {TEST_SPLIT_PATH}")
    if os.path.exists(CACHE_FILE):
        print(f"  [CACHE] Loading pre-extracted Unified features from {CACHE_FILE}")
        data = np.load(CACHE_FILE)
        X_pruned = data['X_train']
        y = data['y_seq_labels']
        source_ids_train = data.get('source_ids', np.array([]))
    else:
        print("  Extracting Train Features (full chain: step1+step2+step3 concatenated)...")
        # Extract all 3 steps and concatenate into a 816-dim chain signature
        # Silently use Parallel
        
        results = Parallel(n_jobs=-1)(
            delayed(_process_single_chain)(cid, chains.get(cid, []), gt_chains, SAMPLES_DIR)
            for cid in train_cids
        )
        
        valid_results = [r for r in results if r is not None]
        X_list = [r[0] for r in valid_results]
        y_list = [r[1] for r in valid_results]
        source_ids_list = [r[2] for r in valid_results]
    
        X = np.vstack(X_list)
        y = np.array(y_list)
        source_ids_train = np.array(source_ids_list)
        
        feat_dim = X.shape[1]
        print(f"  Samples used: {X.shape[0]} | Feature dims (raw): {feat_dim}")
    
        # Pruning removed as Q-table features are now consistently ordered
        X_pruned = X
    
        # Cache Naming Convention Guardrail
        np.savez_compressed(CACHE_FILE, X_train=X_pruned, y_seq_labels=y, source_ids=source_ids_train)
        print(f"  Saved unified cache to {CACHE_FILE}")

    # Train Seq2Seq Model
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, val_idx = next(gss_val.split(X_pruned, y, groups=source_ids_train))
    
    X_tr, X_val = X_pruned[tr_idx], X_pruned[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    base_rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_features=0.2, class_weight="balanced", random_state=42, n_jobs=-1)
    seq_model = ClassifierChain(base_rf, order=[0, 1, 2], cv=5)
    
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

    # Train and Save Surface Model (Single Image Mode)
    # We train this exclusively on the Step 3 features for analyze_image.py portability
    # Step 3 features are the last 272 raw dims (indices 544-815)
    
    X_surf = X_pruned[:, 544:] if X_pruned.shape[1] == 816 else X_pruned
    y_surf = y[:, 2] # Step 3 label
    
    surf_model = RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_features=0.2, class_weight="balanced", random_state=42, n_jobs=-1)
    surf_model.fit(X_surf, y_surf)
    joblib.dump(surf_model, SURF_MODEL_PATH)
    print(f"  Saved single-step Surface Model to {SURF_MODEL_PATH}")

    # Train and Save True Sequence Model (Blind Single Image Mode)
    # This trains a Classifier Chain strictly on Step 3 features to predict the entire 3-step sequence
    # without using intermediate features.
    X_true_seq = X_pruned[:, 544:] if X_pruned.shape[1] == 816 else X_pruned
    TRUE_SEQ_MODEL_PATH = os.path.join(MODELS_DIR, "true_seq_model.joblib")
    
    true_seq_model = ClassifierChain(base_rf, order=[0, 1, 2], cv=5)
    true_seq_model.fit(X_true_seq, y)
    joblib.dump(true_seq_model, TRUE_SEQ_MODEL_PATH)
    print(f"  Saved True blind Sequence Model to {TRUE_SEQ_MODEL_PATH}")

if __name__ == "__main__":
    main()
