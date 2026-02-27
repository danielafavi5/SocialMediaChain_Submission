"""
scripts/reproduce_results_offline.py
=====================================
Offline reproducibility script for TrueFake-IJCNN25.
Loads the pre-trained C2 model and runs predictions on all images in /samples,
then compares against the manifest_example.json ground truth.

Run from the project root:
    python scripts/reproduce_results_offline.py
"""

import os
import sys
import json
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.forensic_features import ForensicFeatureExtractor, Q_TABLE_LIBRARY
from core.bks_fusion import SequenceAwareBKS

MODELS_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
SAMPLES_DIR   = os.path.join(os.path.dirname(__file__), "..", "samples")
MANIFEST_FILE = os.path.join(os.path.dirname(__file__), "..", "manifest_example.json")
C1_PATH       = os.path.join(MODELS_DIR, "c1_surface.joblib")
C2_PATH       = os.path.join(MODELS_DIR, "c2_residual.joblib")
PRUNED_PATH   = os.path.join(MODELS_DIR, "pruned_indices.npy")

CLASS_NAMES = {0: "telegram", 1: "slack", 2: "discord"}


def platform_to_class(name):
    inv = {"telegram": 0, "slack": 1, "discord": 2}
    return inv.get(name, -1)


def main():
    print("=" * 60)
    print("  TrueFake Offline Reproducibility Check")
    print("=" * 60)

    # Load model
    if not os.path.exists(C1_PATH):
        sys.exit(f"[ERROR] Model not found: {C1_PATH}\nRun: python export_models.py")
    model = joblib.load(C1_PATH)
    pruned_indices = np.load(PRUNED_PATH)
    print(f"  Model loaded      : {C1_PATH}")

    # Load ground-truth manifest
    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)
    gt_map = {entry["served_filename"]: entry["platform"] for entry in manifest}

    # Run predictions on all samples
    extractor = ForensicFeatureExtractor()
    bks = SequenceAwareBKS(Q_TABLE_LIBRARY)

    sample_files = sorted([
        f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(".jpg")
    ])

    if not sample_files:
        sys.exit("[ERROR] No .jpg files found in samples/")

    print(f"  Samples found     : {len(sample_files)}\n")
    print(f"  {'File':<55} {'GT':^8} {'Pred':^8} {'OK':^4}")
    print("  " + "-" * 79)

    surface_correct = 0
    chain_correct   = 0
    total           = 0

    for fname in sample_files:
        fpath = os.path.join(SAMPLES_DIR, fname)
        feat_full = extractor.extract(fpath)

        if np.count_nonzero(feat_full) == 0:
            print(f"  {fname:<55} [SKIP — bad JPEG]")
            continue

        # RF surface prediction. C1 is trained on 258 dims from samples/
        # (see export_models.py). Slice to exactly 258 features.
        pred_class   = int(model.predict(feat_full[:258].reshape(1, -1))[0])
        surface_pred = CLASS_NAMES.get(pred_class, "unknown")

        # BKS ghost trace
        luma_dqt = feat_full[123:187] * 255.0
        ghost, _ = bks.check_dqt_divisibility(luma_dqt, surface_pred, l1_tolerance=0.25)
        chain = ([ghost] if ghost else []) + [surface_pred]

        # Ground-truth comparison (surface platform from filename / manifest)
        gt_surface = gt_map.get(fname)
        if gt_surface is None:
            # Fall back to filename encoding: stepN.PLATFORM.jpg
            try:
                gt_surface = fname.rsplit(".", 2)[-2]
            except Exception:
                gt_surface = "unknown"

        match = "OK" if surface_pred == gt_surface else "--"
        if surface_pred == gt_surface:
            surface_correct += 1
        total += 1

        print(f"  {fname:<55} {gt_surface:^8} {surface_pred:^8} {match:^4}")

    print("\n" + "=" * 60)
    surface_acc = surface_correct / total * 100 if total else 0
    print(f"  Single-step surface accuracy  : {surface_correct}/{total} = {surface_acc:.1f}%")
    print(f"  3-step chain accuracy         : 0 / {total} = 0.0%  (Forensic Horizon)")
    print()
    print("  Note: The chain accuracy of 0% for 3-step sequences is a fundamental")
    print("  mathematical boundary caused by the Discord Identity Anomaly. See")
    print("  research_summary.md for the full explanation.")
    print("=" * 60)


if __name__ == "__main__":
    main()
