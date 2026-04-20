"""
bks_fusion.py
=============
This module contains the Sequence-Aware Behavior-Knowledge Space (BKS) Fusion logic.
It resolves conflicts between the primary surface platform classifier (C1 / Q-table match)
and the deep historical residual classifier (C2_Residual).

It implements the 'tracing limit' safeguards and 'Targeted DQT Backtracking'
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
        if isinstance(lib_dqt_or_platform, np.ndarray):
            return self._check_pair(current_dqt, lib_dqt_or_platform, l1_tolerance)
        current_platform = lib_dqt_or_platform
        targets = {"2026 Telegram": "telegram", "2026 Slack": "slack", "2026 Discord": "discord"}
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
        if current_dqt.shape != lib_dqt.shape:
            return False, None
        if np.sum(lib_dqt == 1) > 32:
            return False, None
        lib_safe = np.where(lib_dqt == 0, 1e-6, lib_dqt).astype(float)
        ratio    = current_dqt.astype(float) / lib_safe
        nearest  = np.rint(ratio)
        mean_error   = float(np.mean(np.abs(ratio - nearest)))
        mean_nearest = float(np.mean(np.abs(nearest)))
        if mean_error < l1_tolerance:
            return True, {
                "ratio":        nearest.astype(int).tolist(),
                "mean_error":   mean_error,
                "mean_nearest": mean_nearest,
            }
        return False, None

    def fuse_sequence(self, surface_predictions, residual_predictions, dqt_arrays):
        assert len(surface_predictions) == 3
        
        reconstructed = ["", "", ""]
        
        for step_index in range(3):
            current_surface = surface_predictions[step_index]
            rf_prev_pred = residual_predictions[step_index] if step_index > 0 else None
            current_dqt = dqt_arrays[step_index]
            
            reconstructed[step_index] = current_surface
            
            if step_index == 0:
                continue
                
            div_match, div_err = self.check_dqt_divisibility(current_dqt, current_surface)
            
            prev_pred_final = rf_prev_pred
            
            if div_match:
                prev_pred_final = div_match
                
            reconstructed[step_index - 1] = prev_pred_final
            
        return reconstructed
            
        return reconstructed
