import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score
from joblib import Parallel, delayed

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.forensic_features import ForensicFeatureExtractor

SAMPLES_DIR = os.path.join(BASE_DIR, "results_2026", "results_2026")
CACHE_FILE = os.path.join(BASE_DIR, "unified_sequence_cache_2026.npz")
CLASS_MAP = {"telegram": 0, "slack": 1, "discord": 2}

def _process_single_chain(cid, chain_files, samples_dir):
    from core.forensic_features import ForensicFeatureExtractor
    ext = ForensicFeatureExtractor()
    
    step1_file = next((f for f in chain_files if ".step1." in f), None)
    step2_file = next((f for f in chain_files if ".step2." in f), None)
    step3_file = next((f for f in chain_files if ".step3." in f), None)
    
    if not (step1_file and step2_file and step3_file): 
        return None
        
    seq = [None, None, None]
    seq[0] = step1_file.split(".step1.")[1].split(".jpg")[0]
    seq[1] = step2_file.split(".step2.")[1].split(".jpg")[0]
    seq[2] = step3_file.split(".step3.")[1].split(".jpg")[0]

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
        
    source_img_id = step1_file.split(".chain_")[0]
    
    return np.concatenate(vecs), lbls, source_img_id

def main():
    print(f"Building Model from {SAMPLES_DIR}...")

    jpgs = sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(".jpg"))
    chains = {}
    for f in jpgs:
        try:
            cid = f.split(".chain_")[1].split(".step")[0]
            if cid not in chains: chains[cid] = []
            chains[cid].append(f)
        except Exception:
            pass

    chain_ids = sorted(list(chains.keys()))
    
    if os.path.exists(CACHE_FILE):
        print(f"  [CACHE] Loading pre-extracted features from {CACHE_FILE}")
        data = np.load(CACHE_FILE, allow_pickle=True)
        X = data['X_train']
        y = data['y_seq_labels']
        source_ids = data['source_ids']
    else:
        print("  Extracting Features (this might take a while)...")
        results = Parallel(n_jobs=-1)(
            delayed(_process_single_chain)(cid, chains.get(cid, []), SAMPLES_DIR)
            for cid in chain_ids
        )
        
        valid_results = [r for r in results if r is not None]
        X = np.vstack([r[0] for r in valid_results])
        y = np.array([r[1] for r in valid_results])
        source_ids = np.array([r[2] for r in valid_results])
        
        np.savez_compressed(CACHE_FILE, X_train=X, y_seq_labels=y, source_ids=source_ids)
        print(f"  Saved cache to {CACHE_FILE}")

    print(f"  Total Chains Extracted: {X.shape[0]}")
    
    print("  Applying GroupShuffleSplit based on source image...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=source_ids))
    
    X_tr, X_val = X[train_idx], X[test_idx]
    y_tr, y_val = y[train_idx], y[test_idx]

    base_rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_features=0.2, class_weight="balanced", random_state=42, n_jobs=-1)
    seq_model = ClassifierChain(base_rf, order=[0, 1, 2], cv=5)
    
    seq_model.fit(X_tr, y_tr)
    y_pred_val = seq_model.predict(X_val)
    
    f1_step1 = f1_score(y_val[:, 0], y_pred_val[:, 0], average="macro")
    f1_step2 = f1_score(y_val[:, 1], y_pred_val[:, 1], average="macro")
    f1_step3 = f1_score(y_val[:, 2], y_pred_val[:, 2], average="macro")
    print(f"  Validation macro-F1: [Step1: {f1_step1:.3f}, Step2: {f1_step2:.3f}, Step3: {f1_step3:.3f}]")
    
    seq_acc = np.mean(np.all(y_pred_val == y_val, axis=1))
    print(f"  Validation Exact Sequence Match: {seq_acc*100:.1f}%")

    import joblib
    model_path = os.path.join(BASE_DIR, "models", "seq_model_2026.joblib")
    joblib.dump(seq_model, model_path)
    print(f"  Saved end-to-end 2026 Sequence Model to {model_path}")

if __name__ == "__main__":
    main()
