"""
export_models.py
================
Trains the C1 (surface RF) and C2 (residual RF) classifiers and saves them
to models/ as .joblib artifacts for offline reproducibility.

- C1 is trained on samples/ with the current 258-dim feature extractor.
  This ensures C1's expected input dimension is always consistent with inference.
- C2 is trained on the Phase 1 historical cache (mixed 2024+2026, adversarial pruning).

Run once from the project root:
    python export_models.py
"""

import os
import sys
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib

sys.path.insert(0, os.path.dirname(__file__) or '.')
from core.forensic_features import ForensicFeatureExtractor

CACHE_FILE   = "phase7_features_cache.npz"
MODELS_DIR   = "models"
C1_PATH      = os.path.join(MODELS_DIR, "c1_surface.joblib")
C2_PATH      = os.path.join(MODELS_DIR, "c2_residual.joblib")
PRUNED_IDX   = os.path.join(MODELS_DIR, "pruned_indices.npy")

# Adversarial pruning: indices of year-discriminative drift features to remove from C2
KS_TOP_DRIFT_INDICES = [225, 224, 138, 139, 145]

# Map raw label strings to 3 logical classes (same mapping as original training)
LOGICAL_MAP = {
    "telegram": 0, "slack": 1, "discord": 2,
    "2024 Facebook": 0, "2026 Telegram": 0,
    "2024 Flickr":   1, "2026 Slack":    1,
    "2024 Twitter":  2, "2026 Discord":  2,
}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # C1: Surface classifier ─ trained live from samples/ with current extractor
    # -----------------------------------------------------------------------
    print("Building C1 from samples/ (live extraction, 258 dims)...")
    ext = ForensicFeatureExtractor()
    CLASS_MAP   = {"telegram": 0, "slack": 1, "discord": 2}
    SAMPLES_DIR = os.path.join(".", "samples")
    MANIFEST    = os.path.join(".", "manifest.json")

    with open(MANIFEST) as mf:
        gt_map = {e["served_filename"]: e["platform"] for e in json.load(mf)}

    X1_list, y1_list = [], []
    jpgs = sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(".jpg"))
    
    # Create an 80/20 train-test split by CHAIN ID to preserve 3-step sequences
    chains = {}
    for f in jpgs:
        try:
            cid = f.split(".chain_")[1].split(".step")[0]
            if cid not in chains: chains[cid] = []
            chains[cid].append(f)
        except: pass
    
    chain_ids = sorted(list(chains.keys()))
    train_cids, test_cids = train_test_split(chain_ids, test_size=0.2, random_state=42)
    
    train_files = [f for cid in train_cids for f in chains[cid]]
    test_files  = [f for cid in test_cids for f in chains[cid]]
    
    # Save the test split for offline evaluation
    TEST_SPLIT_PATH = os.path.join(".", "samples_test_split.json")
    with open(TEST_SPLIT_PATH, "w") as f:
        json.dump(test_files, f, indent=2)
    print(f"  Saved test split to {TEST_SPLIT_PATH}")

    for i, fname in enumerate(train_files, 1):
        fvec = ext.extract(os.path.join(SAMPLES_DIR, fname))
        if not np.count_nonzero(fvec):
            continue
        gt = gt_map.get(fname, fname.rsplit(".", 2)[-2])
        lbl = CLASS_MAP.get(gt)
        if lbl is None:
            continue
        X1_list.append(fvec)
        y1_list.append(lbl)
        if i % 100 == 0:
            print(f"  Extracted {i}/{len(jpgs)}...")

    X1 = np.vstack(X1_list)
    y1 = np.array(y1_list)
    print(f"  Samples used  : {X1.shape[0]}  |  Feature dims : {X1.shape[1]}")

    # Here we can just directly train on X1/y1 and use CV score or typical random split
    X1_tr, X1_val, y1_tr, y1_val = train_test_split(
        X1, y1, test_size=0.2, random_state=42, stratify=y1
    )
    c1 = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                class_weight="balanced", random_state=42, n_jobs=1)
    c1.fit(X1_tr, y1_tr)
    c1_f1 = f1_score(y1_val, c1.predict(X1_val), average="macro")
    print(f"\nC1 (surface, samples/ train split) validation macro-F1: {c1_f1:.4f}")
    
    # Fit on all training data eventually, but we leave as is for now:
    c1.fit(X1, y1)
    joblib.dump(c1, C1_PATH)
    print(f"  Saved -> {C1_PATH}")

    # -----------------------------------------------------------------------
    # C2: Residual classifier ─ trained on Phase 7 augmented cache
    #     (adversarial-pruned, mixed 2024+2026)
    # -----------------------------------------------------------------------
    print(f"\nLoading Phase 7 augmented cache: {CACHE_FILE}")
    cache = np.load(CACHE_FILE, allow_pickle=True)
    X_aug      = cache["X_aug"]
    labels_aug = cache["labels_aug"]
    
    # Optional fallback for legacy caches
    cids_aug = cache.get("cids_aug", np.array(["unknown"] * len(labels_aug)))

    # Fetch holdout chains from the JSON split
    test_chains = set()
    with open(TEST_SPLIT_PATH, 'r') as f:
        test_fs = json.load(f)
        for tf in test_fs:
            try:
                test_chains.add(tf.split('.chain_')[1].split('.step')[0])
            except Exception: pass

    # Strict Parent-Child Leakage filter
    train_mask = np.array([cid not in test_chains for cid in cids_aug])
    
    X_aug = X_aug[train_mask]
    labels_aug = labels_aug[train_mask]
    
    y_platform = np.array([LOGICAL_MAP.get(lbl, -1) for lbl in labels_aug])

    feat_dim = X_aug.shape[1]
    print(f"  Total samples : {X_aug.shape[0]} (Removed {sum(~train_mask)} test leaks) | Feature dims : {feat_dim}")

    # Adversarial pruning
    valid_indices = [i for i in range(feat_dim) if i not in KS_TOP_DRIFT_INDICES]
    X_pruned = X_aug[:, valid_indices]
    np.save(PRUNED_IDX, np.array(valid_indices))
    print(f"  Pruned to     : {X_pruned.shape[1]} dimensions")

    X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
        X_pruned, y_platform, test_size=0.2, random_state=42, stratify=y_platform
    )
    c2 = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                class_weight="balanced", random_state=42, n_jobs=1)
    c2.fit(X_tr2, y_tr2)
    c2_f1 = f1_score(y_val2, c2.predict(X_val2), average="macro")
    print(f"\nC2 (residual, mixed)    macro-F1 : {c2_f1:.4f}")
    joblib.dump(c2, C2_PATH)
    print(f"  Saved -> {C2_PATH}")
    print("\nModel export complete.")


if __name__ == "__main__":
    main()
