"""
phase8_evaluation.py
====================
Phase 8: Recursive DQT Backtracking (Forensic Horizon)

Evaluates the 100 chaining test set using the Augmented Random Forest model,
but introduces an explicit deterministic heuristic: DQT Divisibility Checks.

By calculating if the current image's DQT is a mathematical multiple (ratio L1 distance)
of historical standard tables, we can identify "Ghost Factors" and break the loop
collapse that affects pure feature-space Random Forests.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from forensic_features import ForensicFeatureExtractor, FEATURE_DIM, Q_TABLE_LIBRARY

CACHE_BASE = "phase5_features_cache.npz"
CACHE_AUG = "phase7_features_cache.npz"
MANIFEST_FILE = "results_2026/manifest.json"

LOGICAL_MAP = {
    "2024 Facebook": 0, "2026 Telegram": 0,
    "2024 Flickr":   1, "2026 Slack":    1,
    "2024 Twitter":  2, "2026 Discord":  2
}
CLASS_TO_PLATFORM = {0: "telegram", 1: "slack", 2: "discord", -1: "UNKNOWN"}
PLATFORM_TO_CLASS = {"telegram": 0, "slack": 1, "discord": 2}

RESIDUAL_INDICES = list(range(123)) + list(range(247, 258))

def load_pipelines():
    print(f"Loading Feature Caches (Base + Augmented)...")
    cache_base = np.load(CACHE_BASE, allow_pickle=True)
    X_base = np.vstack([cache_base[f"mat_{i}"] for i in range(6)])
    y_plat_base = np.array([LOGICAL_MAP[lbl] for lbl in cache_base["labels_all"]])
    
    X_train_f, X_val_f, y_train_f, y_val_f = train_test_split(
        X_base, y_plat_base, test_size=0.1, random_state=42, stratify=y_plat_base
    )
    
    c2_full = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42)
    c2_full.fit(X_train_f, y_train_f)
    
    cache_aug = np.load(CACHE_AUG, allow_pickle=True)
    X_aug = cache_aug["X_aug"]
    y_plat_aug = np.array([PLATFORM_TO_CLASS[p] for p in cache_aug["labels_aug"]])
    
    X_res_combined = np.vstack([X_base[:, RESIDUAL_INDICES], X_aug[:, RESIDUAL_INDICES]])
    y_res_combined = np.concatenate([y_plat_base, y_plat_aug])
    
    c2_resid = RandomForestClassifier(n_estimators=500, max_depth=20, n_jobs=-1, random_state=42)
    c2_resid.fit(X_res_combined, y_res_combined)

    return c2_full, c2_resid

def simulate_inference_c1(true_platform):
    cls = PLATFORM_TO_CLASS[true_platform]
    if np.random.rand() < 0.65: return cls
    return -1 if np.random.rand() < 0.2 else np.random.choice([0,1,2])

def get_chains_from_filenames(target_chains=100):
    import glob
    from collections import defaultdict
    files = glob.glob("results_2026/*.step*.jpg")
    chains_dict = defaultdict(list)
    for f in files:
        fname = os.path.basename(f)
        try:
            base_id = fname.split('.step')[0]
            chains_dict[base_id].append(f)
        except: pass
            
    full_chains = []
    for cid, entries in chains_dict.items():
        if len(entries) >= 3:
            s1 = next((e for e in entries if '.step1.' in e), None)
            s2 = next((e for e in entries if '.step2.' in e), None)
            s3 = next((e for e in entries if '.step3.' in e), None)
            if s1 and s2 and s3:
                def parse_plat(p): return os.path.basename(p).split('.')[3]
                seq = [parse_plat(s1), parse_plat(s2), parse_plat(s3)]
                chain_objs = [
                    {"chain_id": cid, "sequence": seq, "step": 1, "platform": seq[0], "served_filename": os.path.basename(s1)},
                    {"chain_id": cid, "sequence": seq, "step": 2, "platform": seq[1], "served_filename": os.path.basename(s2)},
                    {"chain_id": cid, "sequence": seq, "step": 3, "platform": seq[2], "served_filename": os.path.basename(s3)}
                ]
                full_chains.append(chain_objs)
                if len(full_chains) == target_chains: break
    return full_chains

import configparser as _configparser
import os as _os

_config = _configparser.ConfigParser()
_config.read(_os.path.join(_os.path.dirname(__file__), '..', 'config.example.ini'))
_IDENTITY_QTABLE_ONES_THRESHOLD = int(_config['DEFAULT'].get('IDENTITY_QTABLE_ONES_THRESHOLD', 32))
_MEAN_ERROR_THRESH = float(_config['DEFAULT'].get('MEAN_ERROR_THRESH', 0.25))
_MEAN_NEAREST_MIN = float(_config['DEFAULT'].get('MEAN_NEAREST_MIN', 1.0))


def check_dqt_divisibility(current_dqt: np.ndarray, lib_dqt: np.ndarray):
    """
    Check whether current_dqt is a plausible integer multiple of lib_dqt.

    Applies a general safeguard that skips tables with an excessive number of cells
    equal to 1 (threshold defined by IDENTITY_QTABLE_ONES_THRESHOLD in config). For
    the current 2026 Discord Q-table this threshold is not triggered — Discord is
    instead blocked because its small coefficients do not produce a clean integer ratio
    against real compressed images (mean_error >= MEAN_ERROR_THRESH).
    Uses epsilon (1e-6) instead of 0 for zero-value cells to avoid division errors.

    Args:
        current_dqt: 1-D or 2-D numpy array of the observed Luma quantization table.
        lib_dqt:     1-D or 2-D numpy array of the candidate library table.

    Returns:
        (True,  {'ratio': list[int], 'mean_error': float, 'mean_nearest': float})
            if divisibility criterion is met, otherwise
        (False, None)
    """
    if current_dqt.shape != lib_dqt.shape:
        return False, None
    if np.sum(lib_dqt == 1) > _IDENTITY_QTABLE_ONES_THRESHOLD:
        return False, None
    eps = 1e-6
    lib_safe = np.where(lib_dqt == 0, eps, lib_dqt).astype(float)
    ratio = current_dqt.astype(float) / lib_safe
    nearest = np.rint(ratio)
    mean_error = float(np.mean(np.abs(ratio - nearest)))
    mean_nearest = float(np.mean(np.abs(nearest)))
    if mean_error < _MEAN_ERROR_THRESH and mean_nearest > _MEAN_NEAREST_MIN:
        return True, {
            'ratio': nearest.astype(int).tolist(),
            'mean_error': mean_error,
            'mean_nearest': mean_nearest,
        }
    return False, None


def main():
    print("="*65)
    print(" PHASE 8: RECURSIVE DQT BACKTRACKING")
    print("="*65)

    c2_full, c2_resid = load_pipelines()
    extractor = ForensicFeatureExtractor()
    chains = get_chains_from_filenames(100)
    
    perfect_chains = 0
    stats_total = 0
    y_true_chains = []
    y_pred_chains = []
    
    # Store proof of DQT backtracking breaking loops
    conflict_logs = []
    recovered_samples = []
    
    print(f"\nEvaluating {len(chains)} Chains with Targeted DQT Overrides...")
    results_dir = os.path.dirname(MANIFEST_FILE)
    
    rescues = 0
    total_chains = 0
    
    for i, chain in enumerate(chains, 1):
        sequence = chain[0]["sequence"]
        
        # Raw RF sequence prediction
        rf_predicted_sequence = ["", "", ""]
        # Heuristic sequence prediction
        heuristic_predicted_sequence = ["", "", ""]
        
        chain_correct = True
        
        for step_idx, entry in enumerate(chain, 1):
            true_plat = entry["platform"]
            served_file = os.path.join(results_dir, entry["served_filename"])
            
            if not os.path.exists(served_file):
                chain_correct = False; continue
                
            feat = extractor.extract(served_file)
            if np.count_nonzero(feat) == 0:
                chain_correct = False; continue
                
            # Raw RF Prediction for previous step
            feat_res = feat[RESIDUAL_INDICES].reshape(1, -1)
            raw_rf_prev_pred = CLASS_TO_PLATFORM.get(c2_resid.predict(feat_res)[0], "UNKNOWN")
            
            # Surface Prediction (current step)
            q_dists = feat[252:258]
            surface_idx = np.argmin(q_dists[3:6]) + 3
            current_surface = ["telegram", "slack", "discord"][surface_idx - 3]
            
            # --- Phase 8 Divisibility Heuristic ---
            current_dqt = feat[123:187] * 255.0
            div_match, div_error, ratio_arr = check_dqt_divisibility(current_dqt, current_surface)
            
            prev_pred_name_rf = raw_rf_prev_pred
            prev_pred_name_heuristic = raw_rf_prev_pred
            
            if step_idx == 1:
                rf_predicted_sequence[0] = current_surface
                heuristic_predicted_sequence[0] = current_surface
                
            elif step_idx == 2:
                rf_predicted_sequence[1] = current_surface
                heuristic_predicted_sequence[1] = current_surface
                
                # BKS Rule Update: If surface is Discord, trust the Telegram/Slack ghost.
                if current_surface == "discord" and div_match:
                    prev_pred_name_heuristic = div_match
                    if len(recovered_samples) < 5:
                        recovered_samples.append({
                            "chain": "->".join(sequence),
                            "step": 2,
                            "rf_pred": raw_rf_prev_pred,
                            "dqt_override": div_match,
                            "div_error": div_error,
                            "ratio_sample": ratio_arr[:8]
                        })
                        
                rf_predicted_sequence[0] = prev_pred_name_rf
                heuristic_predicted_sequence[0] = prev_pred_name_heuristic
                
            elif step_idx == 3:
                rf_predicted_sequence[2] = current_surface
                heuristic_predicted_sequence[2] = current_surface
                
                if current_surface == "discord" and div_match:
                    prev_pred_name_heuristic = div_match
                    
                rf_predicted_sequence[1] = prev_pred_name_rf
                heuristic_predicted_sequence[1] = prev_pred_name_heuristic
                
        # Calculate Rescue
        rf_correct_steps = sum([1 for a, b in zip(sequence, rf_predicted_sequence) if a == b])
        heuristic_correct_steps = sum([1 for a, b in zip(sequence, heuristic_predicted_sequence) if a == b])
        
        # Did it move from 0% correct to >= 2/3 correct?
        if rf_correct_steps == 0 and heuristic_correct_steps >= 2:
            rescues += 1
            conflict_logs.append(f"RESCUED: { '->'.join(sequence) } (RF: {rf_correct_steps}/3 -> DQT: {heuristic_correct_steps}/3)")
            
        if heuristic_correct_steps == 3:
            perfect_chains += 1
            
        stats_total += 1
        
        true_str = "->".join(sequence)
        pred_str = "->".join(heuristic_predicted_sequence)
        y_true_chains.append(true_str)
        y_pred_chains.append(pred_str)
        
        if i % 20 == 0:
            print(f"  Processed {i}/{len(chains)} chains...")
            
    print("\n" + "="*65)
    accuracy = (perfect_chains / stats_total) * 100
    rescue_rate = (rescues / stats_total) * 100
    print(f" FINAL PHASE 8.1 DQT BACKTRACKING ACCURACY: {perfect_chains}/{stats_total} ({accuracy:.1f}%)")
    print(f" DQT RESCUE RATE (Chains moved from 0/3 -> >=2/3 correct): {rescues}/{stats_total} ({rescue_rate:.1f}%)")
    print("="*65)
    
    print("\n[DIVISIBILITY TRACE SAMPLES] (Chains broken from RF Loops):")
    for r in recovered_samples[:3]:
        print(f"  Chain: {r['chain']} (Backtracking from Step {r['step']})")
        print(f"   - RF Stuck Loop Predicts Prev: {r['rf_pred']}")
        print(f"   - DQT Divisibility Override : {r['dqt_override']} (L1 Error: {r['div_error']:.4f})")
        print(f"   - Ratio Top-Left 8x8 ACs : {np.round(r['ratio_sample'], 2)}")
        print()
        
    print("\n[CONFLICT LOG] Successfully rescued chains:")
    for c in list(set(conflict_logs))[:5]:
        print(f"  {c}")
        
    print("\n[FINAL CONFUSION MATRIX]:")
    labels = sorted(list(set(y_true_chains + y_pred_chains)))
    if labels:
        cm = confusion_matrix(y_true_chains, y_pred_chains, labels=labels)
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        print(df_cm)

if __name__ == "__main__":
    main()
