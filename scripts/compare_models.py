import os
import sys
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.forensic_features import ForensicFeatureExtractor

CACHE_2026 = os.path.join(BASE_DIR, "unified_sequence_cache_2026.npz")
TEST_SPLIT_PATH = os.path.join(BASE_DIR, "samples_test_split.json")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
MANIFEST_PATH = os.path.join(BASE_DIR, "manifest.json")
SEQ_MODEL_PATH = os.path.join(BASE_DIR, "models", "seq_model.joblib")
CLASSES = ["telegram", "slack", "discord"]
CLASS_MAP = {"telegram": 0, "slack": 1, "discord": 2}

def main():
    print("=== True Holdout Model Comparison ===")
    
    # 1. Load True Test Set defined in samples_test_split.json
    with open(TEST_SPLIT_PATH, "r") as f:
        test_files = json.load(f)
        
    with open(MANIFEST_PATH, "r") as mf:
        manifest = json.load(mf)
        
    gt_chains = {entry["chain_id"]: entry.get("sequence", []) for entry in manifest if entry.get("chain_id") and len(entry.get("sequence", [])) == 3}
    
    test_chains = {}
    for fname in test_files:
        try:
            cid = fname.split(".chain_")[1].split(".step")[0]
            if cid not in test_chains:
                test_chains[cid] = []
            test_chains[cid].append(fname)
        except Exception:
            pass

    test_source_imgs = set()
    for e in manifest:
        if e.get("chain_id") in test_chains:
            orig = e.get("orig_image", "").replace(".jpg", "")
            if orig:
                test_source_imgs.add(orig)

    print(f"Extracting features for the {len(test_chains)} test chains...")
    extractor = ForensicFeatureExtractor()
    X_test_list = []
    y_test_list = []
    
    for cid, fnames in test_chains.items():
        step1 = next((f for f in fnames if ".step1." in f), None)
        step2 = next((f for f in fnames if ".step2." in f), None)
        step3 = next((f for f in fnames if ".step3." in f), None)
        
        true_seq = gt_chains.get(cid)
        if not (step1 and step2 and step3) or not true_seq: continue
            
        vecs = []
        for fname in [step1, step2, step3]:
            vecs.append(extractor.extract(os.path.join(SAMPLES_DIR, fname)))
        
        X_test_list.append(np.concatenate(vecs))
        y_test_list.append([CLASS_MAP[true_seq[0]], CLASS_MAP[true_seq[1]], CLASS_MAP[true_seq[2]]])

    X_test = np.vstack(X_test_list)
    y_test = np.array(y_test_list)
    
    # 2. Evaluate original seq_model.joblib
    orig_model = joblib.load(SEQ_MODEL_PATH)
    y_pred_orig = orig_model.predict(X_test)
    seq_acc_orig = np.mean(np.all(y_pred_orig == y_test, axis=1))
    
    print("\n--- Original Model (Trained on Samples) ---")
    print(f"  Exact Sequence Match: {seq_acc_orig*100:.1f}%")
    
    # 3. Train and evaluate 2026 model
    data_2026 = np.load(CACHE_2026)
    X_2026_full = data_2026['X_train']
    y_2026_full = data_2026['y_seq_labels']
    source_ids_2026 = data_2026['source_ids']
    
    safe_indices = [i for i, sid in enumerate(source_ids_2026) if sid not in test_source_imgs]
    X_2026_train = X_2026_full[safe_indices]
    y_2026_train = y_2026_full[safe_indices]
    
    print(f"\nTraining 2026 Model (on {len(safe_indices)} safe chains)...")
    base_rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_features=0.2, class_weight="balanced", random_state=42, n_jobs=-1)
    new_model = ClassifierChain(base_rf, order=[0, 1, 2], cv=5)
    new_model.fit(X_2026_train, y_2026_train)
    
    y_pred_new = new_model.predict(X_test)
    seq_acc_new = np.mean(np.all(y_pred_new == y_test, axis=1))
    
    print("--- 2026 Model ---")
    print(f"  Exact Sequence Match: {seq_acc_new*100:.1f}%")

if __name__ == "__main__":
    main()
