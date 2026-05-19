"""
analyze_image.py
================
Forensic platform predictor for a single JPEG image.

Pipeline:
  1. Parse --image argument.
  2. Resolve model paths relative to this script (no hard-coded absolute paths).
  3. Load the pre-trained Surface Platform Classifier and pruned-feature index.
  4. Extract the 272-dim structural feature vector from the image.
  5. Predict the surface platform from the feature vector.
  6. Run BKS DQT divisibility trace to detect a prior-platform ghost.
  7. Print results; exit with code 0 on success, 1 on any failure.

Usage:
    python analyze_image.py --image samples/test.jpg
    python analyze_image.py --image samples/test.jpg --json
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Luma DQT occupies feature indices 123-187 (64 coefficients, normalised to [0,1]).
LUMA_DQT_SLICE = slice(123, 187)
LUMA_DQT_SCALE = 255.0          # de-normalise to original [1-255] range

CLASS_NAMES = {0: "telegram", 1: "slack", 2: "discord"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _repo_root() -> str:
    """Return the directory that contains this script (repo root)."""
    return os.path.dirname(os.path.abspath(__file__))


def _resolve(*parts: str) -> str:
    """Build an absolute path relative to the repo root."""
    return os.path.join(_repo_root(), *parts)


def _fail(message: str, hint: str = "") -> None:
    """Print an error message and exit with code 1."""
    print(f"Error: {message}", file=sys.stderr)
    if hint:
        print(f"Hint:  {hint}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(models_dir: str):
    """
    Load the Unified Sequence Model and the pruned-feature index.

    Args:
        models_dir: Absolute path to the models/ directory.

    Returns:
        Tuple of (seq_model, pruned_indices).

    Raises:
        SystemExit(1) if any model file is missing.
    """
    surf_path    = os.path.join(models_dir, "surface_model.joblib")

    if not os.path.isfile(surf_path):
        _fail(
            "Model file missing: surface_model.joblib",
            "Run  python run_all.py  to train the model and generate all artifacts.",
        )

    surf_model = joblib.load(surf_path)
    return surf_model


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(image_path: str):
    """
    Extract the 272-dim structural feature vector from a JPEG file.

    Args:
        image_path: Absolute or relative path to the JPEG image.

    Returns:
        np.ndarray of shape (272,).
    """
    if not os.path.isfile(image_path):
        _fail(f"Image file not found: '{image_path}'")

    # Import here so that sys.path is set once in main(), not at module level.
    from core.forensic_features import ForensicFeatureExtractor  # noqa: PLC0415

    feat = ForensicFeatureExtractor().extract(image_path)
    if np.count_nonzero(feat) == 0:
        _fail(
            "Feature extraction returned all zeros.",
            "Confirm the file is a valid, non-truncated JPEG.",
        )
    return feat


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(feat: np.ndarray, surf_model) -> tuple[str, float]:
    """
    Predict the surface platform from a single image's feature vector.
    Uses the dedicated surface_model.joblib.
    """
    feat_input = feat.reshape(1, -1)
    surface_idx = int(surf_model.predict(feat_input)[0])
    surface = CLASS_NAMES.get(surface_idx, "unknown")

    probas = surf_model.predict_proba(feat_input)[0]
    confidence = float(np.max(probas))

    return surface, confidence


def ghost_trace(feat: np.ndarray, surface_platform: str) -> tuple[str | None, dict | None]:
    """
    Run the BKS DQT divisibility trace for a prior-platform ghost.

    Returns:
        (ghost_platform: str | None, metadata: dict | None)
    """
    from core.bks_fusion import SequenceAwareBKS   # noqa: PLC0415
    from core.forensic_features import Q_TABLE_LIBRARY  # noqa: PLC0415

    bks      = SequenceAwareBKS(Q_TABLE_LIBRARY)
    luma_dqt = feat[LUMA_DQT_SLICE] * LUMA_DQT_SCALE

    # bks.check_dqt_divisibility returns (bool, dict|None) after the Prompt-2 refactor.
    # Iterate over the library to find the best ghost match.
    best_ghost, best_meta = None, None
    targets = {"telegram": "2026 Telegram", "slack": "2026 Slack"}
    for candidate_platform in ("telegram", "slack"):
        if candidate_platform == surface_platform:
            continue
        lib_key = targets[candidate_platform]
        lib_dqt = Q_TABLE_LIBRARY.get(lib_key)
        if lib_dqt is None:
            continue
        matched, meta = bks.check_dqt_divisibility(luma_dqt, lib_dqt)
        if matched:
            if best_meta is None or meta["mean_error"] < best_meta["mean_error"]:
                best_ghost, best_meta = candidate_platform, meta

    return best_ghost, best_meta


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="TrueFake BKS — forensic platform prediction for JPEG images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the input JPEG image (relative or absolute).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as a JSON object instead of human-readable text.",
    )
    args = parser.parse_args()

    # Resolve repo root and add it to sys.path once, explicitly.
    repo = _repo_root()
    if repo not in sys.path:
        sys.path.insert(0, repo)

    models_dir = _resolve("models")

    # --- Pipeline ---
    surf_model                 = load_model(models_dir)
    feat                       = extract_features(args.image)
    surface, confidence        = predict(feat, surf_model)
    ghost, ghost_meta          = ghost_trace(feat, surface)
    
    # Forensic reconstruction: if ghost found, it is parent. Otherwise unknown parent.
    chain = ([ghost] if ghost else ["unknown"]) + [surface]

    # --- Output ---
    # --- Output ---
    bname = os.path.basename(args.image)
    dirname = os.path.dirname(args.image)
    gt_surface = None
    gt_ghost = None
    gt_chain = None

    if ".chain_" in bname and ".step" in bname:
        try:
            cid = bname.split(".chain_")[1].split(".step")[0]
            step = int(bname.split(".step")[1].split(".")[0])
            gt_surface = bname.split(".")[-2].lower()
            
            # Lookup in manifest
            manifest_path = _resolve("manifest.json")
            if os.path.exists(manifest_path):
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                for entry in manifest:
                    if entry.get("chain_id") == cid and len(entry.get("sequence", [])) == 3:
                        gt_chain = entry["sequence"]
                        break
            
            # If not in manifest, scan directory
            if not gt_chain and os.path.isdir(dirname):
                files = os.listdir(dirname)
                chain_files = [f for f in files if f".chain_{cid}.step" in f]
                s1 = next((f.split(".")[-2].lower() for f in chain_files if ".step1." in f), None)
                s2 = next((f.split(".")[-2].lower() for f in chain_files if ".step2." in f), None)
                s3 = next((f.split(".")[-2].lower() for f in chain_files if ".step3." in f), None)
                if s1 and s2 and s3:
                    gt_chain = [s1, s2, s3]

            if gt_chain:
                if step == 2:
                    gt_ghost = gt_chain[0]
                elif step == 3:
                    gt_ghost = gt_chain[1]

        except Exception:
            pass

    if args.json:
        result = {
            "image":            bname,
            "surface_platform": surface,
            "confidence":       round(confidence, 4),
            "ghost_prior":      ghost,
            "ghost_meta":       ghost_meta,
            "predicted_chain":  chain,
            "gt_surface":       gt_surface,
            "gt_ghost":         gt_ghost,
            "gt_chain":         gt_chain,
            "surface_match":    (surface == gt_surface) if gt_surface else None,
            "ghost_match":      (ghost == gt_ghost) if gt_ghost else None
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"\n  Image            : {bname}")
        print("  " + "-" * 46)
        print(f"  RF prediction    : {surface.upper()}  (confidence {confidence:.1%})")
        if ghost and ghost_meta:
            print(f"  Ghost prior      : {ghost.upper()}  "
                  f"(mean L1 error = {ghost_meta['mean_error']:.4f})")
        else:
            print("  Ghost prior      : none detected")
        print("\n  " + "=" * 46)
        print(f"  Predicted chain  : {' -> '.join(chain)}")
        if gt_surface:
            print(f"  True Surface     : {gt_surface.upper()}")
            print(f"  Status           : [{'SURFACE MATCH' if surface == gt_surface else 'SURFACE MISMATCH'}]")
        if gt_ghost:
            print(f"  True Ghost Prior : {gt_ghost.upper()}")
            print(f"  Status           : [{'GHOST MATCH' if ghost == gt_ghost else 'GHOST MISMATCH'}]")
        if gt_chain:
            print(f"  True Full Chain  : {' -> '.join(gt_chain)}")
        print("  " + "=" * 46 + "\n")


if __name__ == "__main__":
    main()
