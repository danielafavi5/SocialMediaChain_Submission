"""
bks_fusion.py
=============
This module contains the Sequence-Aware Behavior-Knowledge Space (BKS) Fusion logic.
It resolves conflicts between the primary surface platform classifier (C1 / Q-table match)
and the deep historical residual classifier (C2_Residual).

It implements the 'Forensic Horizon' safeguards and 'Targeted DQT Backtracking'
to actively prevent machine learning prediction collapse during deep sequences.
"""

import numpy as np

class SequenceAwareBKS:
    """
    Fuses predictions from multiple layers to reconstruct image sharing chains.
    """
    
    def __init__(self, q_table_library):
        """
        Args:
            q_table_library (dict): Dictionary mapping platform names to standard Q-table numpy arrays.
        """
        self.q_table_library = q_table_library
        
    def check_dqt_divisibility(self, current_dqt, lib_dqt_or_platform, l1_tolerance=0.25):
        """
        Dual-mode divisibility check.

        Low-level mode (two array arguments):
            check_dqt_divisibility(current_dqt, lib_dqt)
            Returns (bool, dict | None) — dict contains 'ratio', 'mean_error', 'mean_nearest'.

        High-level mode (original string argument):
            check_dqt_divisibility(current_dqt, surface_platform_str, l1_tolerance)
            Returns (best_match_platform: str | None, best_error: float | None).
        """
        # Detect call mode by the type of the second argument.
        if isinstance(lib_dqt_or_platform, np.ndarray):
            return self._check_pair(current_dqt, lib_dqt_or_platform, l1_tolerance)
        # --- backward-compatible high-level mode ---
        current_platform = lib_dqt_or_platform
        targets = {"2026 Telegram": "telegram", "2026 Slack": "slack"}
        best_match, best_error = None, float("inf")
        for lib_name, lib_dqt in self.q_table_library.items():
            if lib_name not in targets:
                continue
            target_plat = targets[lib_name]
            if target_plat == current_platform:
                continue
            matched, meta = self._check_pair(current_dqt, lib_dqt, l1_tolerance)
            if matched and meta["mean_error"] < best_error:
                best_error = meta["mean_error"]
                best_match = target_plat
        if best_match is not None:
            return best_match, best_error
        return None, None

    def _check_pair(self, current_dqt: np.ndarray, lib_dqt: np.ndarray,
                    l1_tolerance: float = 0.25):
        """
        Low-level pairwise divisibility check.

        Applies a general safeguard that skips tables with more than 32 cells equal
        to 1. This is a configurable guard for degenerate tables. For the current
        2026 Discord Q-table (which has only 14 ones), this condition is not triggered.
        Discord is instead blocked because its small, clustered coefficients do not
        yield a clean integer ratio against real compressed images — mean_error
        exceeds the l1_tolerance threshold and/or mean_nearest falls at or below 1.0.
        Uses epsilon (1e-6) for zero-cells to avoid division errors.

        Returns:
            (True,  {'ratio': list[int], 'mean_error': float, 'mean_nearest': float})
            (False, None) if the criterion is not met.
        """
        if current_dqt.shape != lib_dqt.shape:
            return False, None
        if np.sum(lib_dqt == 1) > 32:
            return False, None
        lib_safe = np.where(lib_dqt == 0, 1e-6, lib_dqt).astype(float)
        ratio    = current_dqt.astype(float) / lib_safe
        nearest  = np.rint(ratio)
        mean_error   = float(np.mean(np.abs(ratio - nearest)))
        mean_nearest = float(np.mean(np.abs(nearest)))
        if mean_error < l1_tolerance and mean_nearest > 1.0:
            return True, {
                "ratio":        nearest.astype(int).tolist(),
                "mean_error":   mean_error,
                "mean_nearest": mean_nearest,
            }
        return False, None

    def fuse_sequence(self, surface_predictions, residual_predictions, dqt_arrays):
        """
        Reconstructs a 3-step sequence using the BKS fallback logic.
        
        Args:
            surface_predictions (list of str): Surface predictions for Step 1, Step 2, Step 3 images.
            residual_predictions (list of str): C2_Residual predictions for the PREVIOUS step.
            dqt_arrays (list of np.ndarray): Extracted 64-dim Luma tables for each step.
            
        Returns:
            list of str: The reconstructed 3-step platform chain sequence.
        """
        assert len(surface_predictions) == 3
        
        reconstructed = ["", "", ""]
        
        for step_index in range(3):
            current_surface = surface_predictions[step_index]
            rf_prev_pred = residual_predictions[step_index] if step_index > 0 else None
            current_dqt = dqt_arrays[step_index]
            
            # 1. Start by trusting the surface prediction for the current step
            reconstructed[step_index] = current_surface
            
            if step_index == 0:
                # Step 1 has no previous history to trace
                continue
                
            # 2. Targeted DQT Backtracking (Heuristic Override)
            div_match, div_err = self.check_dqt_divisibility(current_dqt, current_surface)
            
            prev_pred_final = rf_prev_pred
            
            # Rule: If surface is Discord, trust Telegram/Slack DQT ghosts if found.
            # Discord is excluded from the ghost-candidate set by _check_pair, so any
            # match here comes from a genuine non-Discord prior platform.
            if current_surface == "discord" and div_match:
                prev_pred_final = div_match
                
            # Update the previous sequence slot based on this step's history tracing
            reconstructed[step_index - 1] = prev_pred_final
            
        return reconstructed
