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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.forensic_features import ForensicFeatureExtractor

CACHE_FILE   = "phase1_features_cache.npz"
MODELS_DIR   = "models"
C1_PATH      = os.path.join(MODELS_DIR, "c1_surface.joblib")
C2_PATH      = os.path.join(MODELS_DIR, "c2_residual.joblib")
PRUNED_IDX   = os.path.join(MODELS_DIR, "pruned_indices.npy")

# Adversarial pruning: indices of year-discriminative drift features to remove from C2
KS_TOP_DRIFT_INDICES = [225, 224, 138, 139, 145]

# Map raw label strings to 3 logical classes (same mapping as original training)
LOGICAL_MAP = {
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
    SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
    MANIFEST    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest_example.json")

    with open(MANIFEST) as mf:
        gt_map = {e["served_filename"]: e["platform"] for e in json.load(mf)}

    X1_list, y1_list = [], []
    jpgs = sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(".jpg"))
    for i, fname in enumerate(jpgs, 1):
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

    X1_tr, X1_val, y1_tr, y1_val = train_test_split(
        X1, y1, test_size=0.2, random_state=42, stratify=y1
    )
    c1 = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                class_weight="balanced", random_state=42, n_jobs=1)
    c1.fit(X1_tr, y1_tr)
    c1_f1 = f1_score(y1_val, c1.predict(X1_val), average="macro")
    print(f"\nC1 (surface, samples/) macro-F1 : {c1_f1:.4f}")
    joblib.dump(c1, C1_PATH)
    print(f"  Saved -> {C1_PATH}")

    # -----------------------------------------------------------------------
    # C2: Residual classifier ─ trained on Phase 1 historical cache
    #     (adversarial-pruned, mixed 2024+2026)
    # -----------------------------------------------------------------------
    print(f"\nLoading Phase 1 historical cache: {CACHE_FILE}")
    cache = np.load(CACHE_FILE, allow_pickle=True)
    matrices   = [cache[f"mat_{i}"] for i in range(6)]
    labels_all = cache["labels_all"]
    years_all  = cache["years_all"]
    X_full     = np.vstack(matrices)
    y_platform = np.array([LOGICAL_MAP[lbl] for lbl in labels_all])

    feat_dim = X_full.shape[1]
    print(f"  Total samples : {X_full.shape[0]}  |  Feature dims : {feat_dim}")

    # Adversarial pruning
    valid_indices = [i for i in range(feat_dim) if i not in KS_TOP_DRIFT_INDICES]
    X_pruned = X_full[:, valid_indices]
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
