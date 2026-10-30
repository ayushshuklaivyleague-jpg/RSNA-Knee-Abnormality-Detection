"""
RSNA Knee Abnormality Detection: Rank-Space Ensembling & Blending Engine

Features:
- Rank-transformation per column (percentile rank), matching ROC-AUC rank-invariance
- Global weight optimizer (single alpha parameter across all 12 targets to prevent overfitting)
- Per-target weight optimizer (constrained weights with L2 regularization)
- Out-of-Fold (OOF) blend validation and incremental contribution analysis
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata

from .metrics import TARGET_COLS, evaluate_predictions


def rank_transform(
    preds: np.ndarray,
) -> np.ndarray:
    """
    Applies percentile rank transformation column by column.
    Handles tied values using average rank.
    """
    ranked = np.empty_like(preds, dtype=np.float64)
    n_rows = preds.shape[0]
    for col in range(preds.shape[1]):
        # rankdata returns 1..N
        r = rankdata(preds[:, col], method="average")
        ranked[:, col] = (r - 0.5) / n_rows
    return ranked


class RankEnsemble:
    """
    Ensemble model for multi-target classification evaluated by Macro ROC-AUC.
    Operates in rank-space to reconcile differing probability calibrations.
    """

    def __init__(
        self,
        target_names: Optional[List[str]] = None,
        use_per_target_weights: bool = False,
        l2_reg: float = 0.01,
    ):
        self.target_names = target_names or TARGET_COLS
        self.use_per_target_weights = use_per_target_weights
        self.l2_reg = l2_reg
        self.weights: Optional[np.ndarray] = None
        self.model_names: List[str] = []

    def fit(
        self,
        y_true: Union[pd.DataFrame, np.ndarray],
        model_oof_preds: List[Union[pd.DataFrame, np.ndarray]],
        model_names: Optional[List[str]] = None,
        method: str = "SLSQP",
    ) -> "RankEnsemble":
        """
        Fits optimal weights on OOF predictions.
        
        Args:
            y_true: Ground truth labels (N, 12).
            model_oof_preds: List of M arrays/DataFrames, each (N, 12).
            model_names: Optional identifiers for each model.
            method: Optimization solver algorithm.
        """
        n_models = len(model_oof_preds)
        self.model_names = model_names or [f"Model_{i}" for i in range(n_models)]

        if isinstance(y_true, pd.DataFrame):
            y_true_arr = y_true[self.target_names].to_numpy()
        else:
            y_true_arr = np.asarray(y_true)

        # Convert each model's predictions to rank space
        ranked_preds = []
        for oof in model_oof_preds:
            if isinstance(oof, pd.DataFrame):
                arr = oof[self.target_names].to_numpy()
            else:
                arr = np.asarray(oof)
            ranked_preds.append(rank_transform(arr))

        # Array of shape (M, N, 12)
        stacked = np.stack(ranked_preds, axis=0)
        n_targets = len(self.target_names)

        if not self.use_per_target_weights:
            # 1 global weight vector of size M
            init_weights = np.ones(n_models) / n_models
            bounds = [(0.0, 1.0) for _ in range(n_models)]
            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

            def loss_func(w):
                norm_w = w / (np.sum(w) + 1e-12)
                # Weighted blend: (N, 12)
                blended = np.tensordot(norm_w, stacked, axes=(0, 0))
                eval_res = evaluate_predictions(y_true_arr, blended, self.target_names)
                return -eval_res["macro_auc"]

            res = minimize(
                loss_func,
                init_weights,
                method=method,
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 200, "disp": False},
            )
            opt_w = res.x / np.sum(res.x)
            # Replicate to (n_targets, n_models)
            self.weights = np.tile(opt_w, (n_targets, 1))

        else:
            # Per-target weights: (n_targets, n_models)
            self.weights = np.zeros((n_targets, n_models))
            for t in range(n_targets):
                target_true = y_true_arr[:, t]
                # (M, N)
                target_stacked = stacked[:, :, t]

                init_weights = np.ones(n_models) / n_models
                bounds = [(0.0, 1.0) for _ in range(n_models)]
                constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

                def target_loss(w):
                    norm_w = w / (np.sum(w) + 1e-12)
                    blended_t = np.dot(norm_w, target_stacked)
                    # Evaluate negative AUC + L2 regularization toward uniform
                    from .metrics import compute_target_auc
                    auc = compute_target_auc(target_true, blended_t)
                    if np.isnan(auc):
                        auc = 0.5
                    reg = self.l2_reg * np.sum((norm_w - 1.0 / n_models) ** 2)
                    return -auc + reg

                res = minimize(
                    target_loss,
                    init_weights,
                    method=method,
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 100, "disp": False},
                )
                self.weights[t, :] = res.x / np.sum(res.x)

        return self

    def predict(
        self,
        model_preds: List[Union[pd.DataFrame, np.ndarray]],
    ) -> np.ndarray:
        """
        Blends test predictions across models using the fitted weights in rank-space.
        
        Args:
            model_preds: List of M arrays/DataFrames, each (N_test, 12).
            
        Returns:
            Blended predictions array of shape (N_test, 12) normalized in [0, 1].
        """
        if self.weights is None:
            raise ValueError("Ensemble has not been fitted yet.")

        ranked_preds = []
        for p in model_preds:
            if isinstance(p, pd.DataFrame):
                arr = p[self.target_names].to_numpy()
            else:
                arr = np.asarray(p)
            ranked_preds.append(rank_transform(arr))

        # (M, N_test, 12)
        stacked = np.stack(ranked_preds, axis=0)
        n_targets = len(self.target_names)
        n_samples = stacked.shape[1]

        blended = np.zeros((n_samples, n_targets), dtype=np.float64)
        for t in range(n_targets):
            w = self.weights[t, :]  # shape (M,)
            # stacked[:, :, t] is (M, N_test)
            blended[:, t] = np.dot(w, stacked[:, :, t])

        # Final rank normalization to guarantee calibrated ordering
        return rank_transform(blended)

    def summary_table(self) -> str:
        """Generates markdown table of weights per model and target."""
        if self.weights is None:
            return "Ensemble not fitted."

        lines = [
            "### Blend Weight Distribution",
            "| Target | " + " | ".join(self.model_names) + " |",
            "| :--- | " + " | ".join([":---:" for _ in self.model_names]) + " |",
        ]
        for t, target in enumerate(self.target_names):
            row_weights = " | ".join([f"{self.weights[t, m]:.3f}" for m in range(len(self.model_names))])
            lines.append(f"| {target} | {row_weights} |")

        return "\n".join(lines)
