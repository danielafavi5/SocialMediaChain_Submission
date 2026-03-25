import os
import sys
import json
import numpy as np
import joblib
from collections import defaultdict
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.forensic_features import ForensicFeatureExtractor, Q_TABLE_LIBRARY
from core.bks_fusion import SequenceAwareBKS

MODELS_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
SAMPLES_DIR   = os.path.join(os.path.dirname(__file__), "..", "samples")
MANIFEST_FILE = os.path.join(os.path.dirname(__file__), "..", "manifest.json")
SPLIT_FILE    = os.path.join(os.path.dirname(__file__), "..", "samples_test_split.json")

C1_PATH       = os.path.join(MODELS_DIR, "c1_surface.joblib")
C2_PATH       = os.path.join(MODELS_DIR, "c2_residual.joblib")
PRUNED_PATH   = os.path.join(MODELS_DIR, "pruned_indices.npy")

CLASS_NAMES = {0: "telegram", 1: "slack", 2: "discord"}
PLATFORM_TO_CLASS = {"telegram": 0, "slack": 1, "discord": 2}

def main():
    print("=" * 60)
    print("  TrueFake Offline Reproducibility Check")
    print("=" * 60)

    # 1. Load Models & Pruned Indices
    if not os.path.exists(C1_PATH) or not os.path.exists(C2_PATH):
        sys.exit(f"[ERROR] Models not found.\nRun: python export_models.py")
    
    c1_model = joblib.load(C1_PATH)
    c2_model = joblib.load(C2_PATH)
    pruned_indices = np.load(PRUNED_PATH)
    print(f"  Models loaded     : {C1_PATH}, {C2_PATH}")

    # 2. Load Ground Truth Manifest
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE) as f:
            manifest = json.load(f)
        # gt_map: filename -> platform
        gt_map = {entry["served_filename"]: entry["platform"] for entry in manifest}
        # gt_chains: chain_id -> sequence list
        gt_chains = {entry["chain_id"]: entry.get("sequence", []) for entry in manifest}
    else:
        gt_map = {}
        gt_chains = {}

    # 3. Load Holdout Test Split
    if not os.path.exists(SPLIT_FILE):
        sys.exit(f"[ERROR] Test split not found: {SPLIT_FILE}\nRun: python export_models.py")
    
    with open(SPLIT_FILE) as f:
        test_files = json.load(f)
    print(f"  Holdout files     : {len(test_files)}")

    # 4. Group test files into chains
    chains = defaultdict(list)
    for fname in test_files:
        if ".chain_" in fname:
            base = fname.split('.step')[0]
            cid = base.split(".chain_")[-1]
            chains[cid].append(fname)

    # Find full 3-step chains
    full_chains = []
    for cid, files in chains.items():
        if len(files) >= 3:
            s1 = next((f for f in files if ".step1." in f), None)
            s2 = next((f for f in files if ".step2." in f), None)
            s3 = next((f for f in files if ".step3." in f), None)
            if s1 and s2 and s3:
                full_chains.append((cid, [s1, s2, s3]))

    print(f"  Full chains to evaluate: {len(full_chains)}\n")
    print(f"  {'Chain ID':<20} {'GT Seq':<30} {'Pred Seq':<30} {'Rescue?':<10}")
    print("  " + "-" * 90)

    extractor = ForensicFeatureExtractor()
    bks = SequenceAwareBKS(Q_TABLE_LIBRARY)

    y_true_c1, y_pred_c1 = [], []
    y_true_c2, y_pred_c2 = [], []
    
    chains_correct = 0
    rf_chains_correct = 0
    rescues = 0

    for cid, fnames in full_chains:
        surf_preds = []
        resid_preds = []
        dqt_arrays = []
        true_seq = []

        # Extract features for all steps
        for step_idx, fname in enumerate(fnames):
            fpath = os.path.join(SAMPLES_DIR, fname)
            feat_full = extractor.extract(fpath)
            
            # Surface GT
            gt_surface = gt_map.get(fname, fname.rsplit(".", 2)[-2])
            true_seq.append(gt_surface)
            y_true_c1.append(gt_surface)
            
            # --- C1 Surface Prediction ---
            c1_feat = feat_full[:258].reshape(1, -1)
            pred_c1_class = int(c1_model.predict(c1_feat)[0])
            c1_str = CLASS_NAMES.get(pred_c1_class, "unknown")
            surf_preds.append(c1_str)
            y_pred_c1.append(c1_str)
            
            # --- C2 Residual Prediction ---
            # The Residual GT is the prior platform (step_idx - 1)
            # If step 1, there is no prior platform theoretically, but we extract anyway
            c2_feat = feat_full[pruned_indices].reshape(1, -1)
            pred_c2_class = int(c2_model.predict(c2_feat)[0])
            c2_str = CLASS_NAMES.get(pred_c2_class, "unknown")
            resid_preds.append(c2_str)
            
            if step_idx > 0:
                y_true_c2.append(true_seq[step_idx - 1])
                y_pred_c2.append(c2_str)
                
            # DQT extraction
            dqt = feat_full[123:187] * 255.0
            dqt_arrays.append(dqt)

        # Apply BKS Fusion
        fused_seq = bks.fuse_sequence(surf_preds, resid_preds, dqt_arrays)
        
        # Determine 3-step accuracy logic
        is_fused_correct = (fused_seq == true_seq)
        is_rf_correct = (surf_preds == true_seq)
        
        if is_fused_correct: chains_correct += 1
        if is_rf_correct: rf_chains_correct += 1
        
        rescue_str = ""
        if (not is_rf_correct) and is_fused_correct:
            rescues += 1
            rescue_str = "RESCUE"
            
        print(f"  {cid[:18]:<20} {'>'.join(true_seq):<30} {'>'.join(fused_seq):<30} {rescue_str:<10}")

    print("\n" + "=" * 60)
    # Calculate Macro-F1
    c1_f1 = f1_score(y_true_c1, y_pred_c1, average="macro") if y_true_c1 else 0.0
    c2_f1 = f1_score(y_true_c2, y_pred_c2, average="macro") if y_true_c2 else 0.0
    
    # Using 0.0% for RF Chain Accuracy to match paper's boundary statement (or actual if somehow correct)
    chain_acc = (chains_correct / len(full_chains)) * 100 if full_chains else 0
    rf_acc = (rf_chains_correct / len(full_chains)) * 100 if full_chains else 0

    print(f"  C1 surface platform identification (2026) : {c1_f1*100:.1f}% macro-F1")
    print(f"  C2 BKS residual classification (mixed)    : {c2_f1*100:.1f}% macro-F1")
    print(f"  Raw RF 3-step chain accuracy              : {rf_acc:.1f}%  (Forensic Horizon)")
    print(f"  BKS Fused chain accuracy                  : {chain_acc:.1f}%")
    print(f"  BKS DQT Rescued Sequences                 : {rescues}")
    print("=" * 60)

if __name__ == "__main__":
    main()
