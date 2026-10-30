"""
src/correlation_stacker.py
==========================
Pathophysiological Second-Stage Correlation Stacker for RSNA Knee Abnormality Detection.
Exploits anatomical co-occurrence networks (Pivot-shift triad, Degenerative medial wear,
Effusion-synovitis coupling) and DICOM metadata to refine base model predictions.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]


class PathophysiologicalStacker:
    """
    Second-stage stacker combining base prediction vectors with clinical
    interaction terms and scanner physics indicators.
    """

    def __init__(self, labels: List[str] = LABELS, alpha: float = 1.0):
        self.labels = labels
        self.alpha = alpha
        self.models: Dict[str, Ridge] = {}

    def engineer_clinical_features(self, prob_df: pd.DataFrame, meta_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Computes clinical interaction features across all 12 findings.
        """
        X = prob_df[self.labels].copy()

        # 1. Classical Biomechanical & Orthopedic Interaction Terms
        X['pivot_shift_triad'] = X['ACL'] * X['Contusion'] * X['Lateral Meniscus']
        X['degenerative_medial_complex'] = X['Medial Meniscus'] * X['Medial OA']
        X['effusion_synovitis_syndrome'] = X['Effusion'] * X['Synovitis']
        X['baker_cyst_effusion_coupling'] = X["Baker's"] * X['Effusion']
        X['patellofemoral_oa_effusion'] = X['PF OA'] * X['Effusion']
        X['collateral_meniscal_coupling'] = X['MCL'] * X['Medial Meniscus']
        X['bicompartmental_oa'] = X['Medial OA'] * X['Lateral OA']
        X['traumatic_bone_disruption'] = X['Fracture'] * X['Contusion']

        # 2. Scanner & Sequence Metadata (if available)
        if meta_df is not None:
            if 'n_series' in meta_df.columns:
                X['n_series'] = meta_df['n_series'].values
            if 'has_fs' in meta_df.columns:
                X['has_fs'] = meta_df['has_fs'].astype(float).values

        return X

    def fit(self, oof_preds: pd.DataFrame, targets: pd.DataFrame, meta_df: Optional[pd.DataFrame] = None) -> None:
        """
        Fit one L2-regularized regressor per finding on Out-of-Fold predictions.
        """
        X = self.engineer_clinical_features(oof_preds, meta_df)

        for target in self.labels:
            y = pd.to_numeric(targets[target], errors='coerce').to_numpy(dtype=float)
            valid = np.isfinite(y)
            if valid.sum() < 10:
                continue

            model = Ridge(alpha=self.alpha, fit_intercept=True)
            model.fit(X.loc[valid].values, y[valid])
            self.models[target] = model

    def predict(self, test_preds: pd.DataFrame, meta_df: Optional[pd.DataFrame] = None, residual_weight: float = 0.12) -> pd.DataFrame:
        """
        Outputs stacked, correlation-adjusted probabilities for test set.
        Blends base probabilities with the correlation model.
        """
        X_test = self.engineer_clinical_features(test_preds, meta_df)
        stacked_out = test_preds[self.labels].copy()

        for target in self.labels:
            if target in self.models:
                stacked_preds = self.models[target].predict(X_test.values)
                # Residual blend: (1 - w) * base + w * stacked
                adjusted = (1.0 - residual_weight) * test_preds[target] + residual_weight * stacked_preds
                stacked_out[target] = np.clip(adjusted, 0.0001, 0.9999)

        if 'StudyInstanceUID' in test_preds.columns:
            stacked_out.insert(0, 'StudyInstanceUID', test_preds['StudyInstanceUID'])

        return stacked_out
