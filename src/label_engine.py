"""
src/label_engine.py
===================
Gold-Calibrated Noise-Robust Label Engine for RSNA Knee Abnormality Detection.
Uses the 58 expert Gold ground-truth studies to calibrate noisy silver report labels,
estimate per-finding noise transition matrices T_c = P(silver | gold), and apply
loss masking so unmentioned findings do not generate false-negative penalties.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# Clinical empirical base priors
DEFAULT_PRIORS = {
    'ACL': 0.14, 'MCL': 0.05, 'Medial Meniscus': 0.28, 'Lateral Meniscus': 0.12,
    'Medial OA': 0.22, 'Lateral OA': 0.15, 'PF OA': 0.38, 'Effusion': 0.48,
    'Synovitis': 0.18, "Baker's": 0.26, 'Contusion': 0.22, 'Fracture': 0.07
}


class GoldCalibratedLabelEngine:
    """
    Estimates per-finding noise transition matrices T_c = P(silver | gold)
    and computes calibrated soft posterior targets P(gold | silver, text_signals).
    """

    def __init__(self, labels: List[str] = LABELS, smoothing: float = 1e-3):
        self.labels = labels
        self.smoothing = smoothing
        self.transition_matrices: Dict[str, np.ndarray] = {}
        self.gold_priors: Dict[str, float] = DEFAULT_PRIORS.copy()

    def fit(self, gold_df: pd.DataFrame, silver_df: pd.DataFrame) -> None:
        """
        Fit transition matrices using the overlapping 58 Gold studies.
        """
        common_uids = list(set(gold_df.index).intersection(set(silver_df.index)))
        if len(common_uids) == 0:
            print("Notice: No overlapping gold/silver UIDs found; using default empirical calibration.")
            return

        g_sub = gold_df.loc[common_uids]
        s_sub = silver_df.loc[common_uids]

        for c in self.labels:
            if c not in g_sub.columns or c not in s_sub.columns:
                continue
            y_true = pd.to_numeric(g_sub[c], errors='coerce').to_numpy(dtype=float)
            y_noisy = pd.to_numeric(s_sub[c], errors='coerce').to_numpy(dtype=float)

            valid = np.isfinite(y_true) & np.isfinite(y_noisy)
            if valid.sum() < 5:
                continue

            yt_v = y_true[valid].astype(int)
            yn_v = (y_noisy[valid] > 0.5).astype(int)

            self.gold_priors[c] = float(np.clip(np.mean(yt_v), 0.02, 0.98))

            cm = confusion_matrix(yt_v, yn_v, labels=[0, 1]).astype(np.float64) + self.smoothing
            row_sums = cm.sum(axis=1, keepdims=True)
            T = cm / row_sums  # Row 0: P(noisy | gold=0); Row 1: P(noisy | gold=1)
            self.transition_matrices[c] = T

    def compute_soft_targets(
        self,
        silver_matrix: np.ndarray,
        addressed_mask: np.ndarray,
        gold_mask: Optional[np.ndarray] = None,
        gold_values: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies Bayes rule to invert P(noisy | gold) into P(gold | noisy).
        Returns:
            calibrated_targets: Soft probabilities in [0.001, 0.999]
            loss_weights: Confidence weights per label (0.0 for unaddressed/masked)
        """
        N, C = silver_matrix.shape
        targets = np.zeros((N, C), dtype=np.float32)
        weights = np.ones((N, C), dtype=np.float32)

        for c_idx, c in enumerate(self.labels):
            p_g1 = self.gold_priors.get(c, DEFAULT_PRIORS.get(c, 0.20))
            p_g0 = 1.0 - p_g1
            T = self.transition_matrices.get(c, np.array([[0.95, 0.05], [0.10, 0.90]]))

            for i in range(N):
                if not addressed_mask[i, c_idx]:
                    # Unaddressed finding: set to empirical prior with ZERO loss weight (masked!)
                    targets[i, c_idx] = p_g1
                    weights[i, c_idx] = 0.0
                    continue

                s_val = silver_matrix[i, c_idx]
                p_noisy_given_g1 = s_val * T[1, 1] + (1.0 - s_val) * T[1, 0]
                p_noisy_given_g0 = s_val * T[0, 1] + (1.0 - s_val) * T[0, 0]

                posterior_g1 = (p_noisy_given_g1 * p_g1) / max(1e-7, (p_noisy_given_g1 * p_g1 + p_noisy_given_g0 * p_g0))
                targets[i, c_idx] = np.clip(posterior_g1, 0.001, 0.999)

                # Reliability-based weighting
                reliability = float(T[1, 1] * T[0, 0])
                weights[i, c_idx] = np.clip(reliability, 0.5, 1.5)

        # Ground truth gold override with 3.0x loss weight
        if gold_mask is not None and gold_values is not None:
            for i in range(N):
                if gold_mask[i]:
                    targets[i] = np.clip(gold_values[i], 0.0, 1.0)
                    weights[i] = 3.0

        return targets, weights
