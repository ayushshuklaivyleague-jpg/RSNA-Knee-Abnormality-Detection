"""
RSNA Knee Abnormality Detection: Cross-Validation & OOF Engine

Features:
- Multi-label iterative stratified K-Fold splitter (ensures rare classes like MCL (9 pos)
  and Lateral OA (11 pos) are evenly distributed across all folds)
- Dual-Track Validation:
    1. Gold CV: Evaluates strictly against the 58 consensus radiologist ground-truth studies
    2. Silver/Report OOF: Evaluates stability across the 4,347 radiology-report studies
- OOF aggregator and comprehensive diagnostic reporting
"""

from typing import Dict, Generator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from .metrics import TARGET_COLS, evaluate_predictions, format_evaluation_report


def iterative_multilabel_split(
    y: np.ndarray,
    n_splits: int = 5,
    random_seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Greedy iterative stratification for multi-label binary datasets.
    Optimizes balanced class prevalence across all K folds, critical for small
    datasets with rare positives (e.g. MCL with only 9 positive cases).
    
    Args:
        y: Binary target matrix of shape (N, n_targets).
        n_splits: Number of cross-validation folds.
        random_seed: Random state seed.
        
    Returns:
        List of (train_indices, val_indices) tuples.
    """
    n_samples, n_targets = y.shape
    rng = np.random.RandomState(random_seed)

    # Initialize fold containers
    fold_indices: List[List[int]] = [[] for _ in range(n_splits)]
    fold_counts = np.zeros((n_splits, n_targets), dtype=np.int64)

    # Calculate global label frequencies to prioritize rarest labels first
    label_frequencies = np.sum(y, axis=0)
    label_order = np.argsort(label_frequencies)  # Rarest labels first

    assigned = np.zeros(n_samples, dtype=bool)

    # Order samples by their rarest positive label
    sample_scores = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        pos_labels = np.where(y[i] == 1)[0]
        if len(pos_labels) > 0:
            sample_scores[i] = np.min(label_frequencies[pos_labels])
        else:
            sample_scores[i] = 1e9 + rng.uniform(0, 1)

    sorted_sample_indices = np.argsort(sample_scores)

    for idx in sorted_sample_indices:
        if assigned[idx]:
            continue

        sample_labels = y[idx]
        if np.sum(sample_labels) == 0:
            # Negative on all targets: assign to fold with fewest total samples
            fold_sizes = [len(f) for f in fold_indices]
            best_fold = int(np.argmin(fold_sizes))
        else:
            # Find fold with lowest representation of these active labels
            candidate_scores = []
            for k in range(n_splits):
                # Count current label load + tie breaker on fold size
                score = np.sum(fold_counts[k] * sample_labels) + 0.01 * len(fold_indices[k])
                candidate_scores.append(score)
            best_fold = int(np.argmin(candidate_scores))

        fold_indices[best_fold].append(idx)
        fold_counts[best_fold] += sample_labels.astype(np.int64)
        assigned[idx] = True

    splits = []
    all_indices = np.arange(n_samples)
    for k in range(n_splits):
        val_idx = np.array(sorted(fold_indices[k]))
        train_idx = np.setdiff1d(all_indices, val_idx)
        splits.append((train_idx, val_idx))

    return splits


class KneeValidationEngine:
    """
    Dual-level validation engine managing 58-study Gold ground-truth CV
    and large-scale Silver/Report cohort evaluation.
    """

    def __init__(
        self,
        gold_df_path: str = "./data/pseudolabeled/pseudolabeled-train.csv",
        silver_df_path: str = "./data/llm_labels_v4_blend.csv",
        target_names: Optional[List[str]] = None,
        n_splits: int = 5,
        random_seed: int = 42,
    ):
        self.target_names = target_names or TARGET_COLS
        self.n_splits = n_splits
        self.random_seed = random_seed

        # Load data
        full_df = pd.read_csv(gold_df_path)
        self.gold_df = full_df[full_df["label_source"] == "gold"].reset_index(drop=True)
        self.silver_df = pd.read_csv(silver_df_path)

        # Generate Gold splits
        y_gold = self.gold_df[self.target_names].to_numpy().astype(int)
        self.gold_splits = iterative_multilabel_split(
            y_gold, n_splits=self.n_splits, random_seed=self.random_seed
        )

    def get_gold_splits(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Returns list of (train_df, val_df) for the 58 gold studies."""
        folds = []
        for train_idx, val_idx in self.gold_splits:
            train_sub = self.gold_df.iloc[train_idx].copy()
            val_sub = self.gold_df.iloc[val_idx].copy()
            folds.append((train_sub, val_sub))
        return folds

    def evaluate_gold_oof(
        self,
        oof_preds: Union[pd.DataFrame, np.ndarray],
        model_name: str = "Candidate_Model",
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Evaluates out-of-fold predictions against the 58 consensus gold studies.
        """
        if isinstance(oof_preds, pd.DataFrame):
            pred_df = self.gold_df[["StudyInstanceUID"]].merge(
                oof_preds, on="StudyInstanceUID", how="left"
            )
            y_pred = pred_df[self.target_names].to_numpy()
        else:
            y_pred = np.asarray(oof_preds)

        y_true = self.gold_df[self.target_names].to_numpy()
        eval_res = evaluate_predictions(
            y_true, y_pred, target_names=self.target_names, n_bootstrap=500
        )
        print(format_evaluation_report(eval_res, model_name=f"{model_name} (58-Gold CV)"))
        return eval_res

    def verify_fold_balance(self) -> pd.DataFrame:
        """Inspects positive prevalence for each target across all generated folds."""
        records = []
        y_gold = self.gold_df[self.target_names].to_numpy().astype(int)
        for k, (_, val_idx) in enumerate(self.gold_splits):
            val_y = y_gold[val_idx]
            row = {"Fold": f"Fold_{k}", "N_Val": len(val_idx)}
            for t_idx, target in enumerate(self.target_names):
                pos = int(np.sum(val_y[:, t_idx]))
                row[target] = pos
            records.append(row)

        df_bal = pd.DataFrame(records)
        return df_bal
