"""
RSNA Knee Abnormality Detection: Model Diagnostics & Diversity Suite

Calculates:
- Target-by-target model prediction rank correlations (Spearman rho)
- Residual disagreement & error complementarity
- Identification of target specialists (which model leads which abnormality)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .metrics import TARGET_COLS, compute_target_auc


def compute_model_correlations(
    preds_dict: Dict[str, Union[pd.DataFrame, np.ndarray]],
    target_names: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Computes pairwise Spearman rank correlation matrices across models for each target.
    
    Args:
        preds_dict: Mapping of model_name -> predictions (N, 12).
        target_names: List of targets.
        
    Returns:
        Dict mapping target_name -> DataFrame (n_models x n_models correlation matrix).
    """
    targets = target_names or TARGET_COLS
    model_names = list(preds_dict.keys())
    n_models = len(model_names)

    # Standardize to numpy arrays
    arr_dict = {}
    for name, p in preds_dict.items():
        if isinstance(p, pd.DataFrame):
            arr_dict[name] = p[targets].to_numpy()
        else:
            arr_dict[name] = np.asarray(p)

    corr_matrices: Dict[str, pd.DataFrame] = {}
    for t_idx, target in enumerate(targets):
        matrix = np.ones((n_models, n_models))
        for i in range(n_models):
            for j in range(i + 1, n_models):
                pred_i = arr_dict[model_names[i]][:, t_idx]
                pred_j = arr_dict[model_names[j]][:, t_idx]
                corr, _ = spearmanr(pred_i, pred_j)
                matrix[i, j] = corr
                matrix[j, i] = corr
        corr_matrices[target] = pd.DataFrame(
            matrix, index=model_names, columns=model_names
        )

    return corr_matrices


def compute_target_specialists(
    y_true: Union[pd.DataFrame, np.ndarray],
    preds_dict: Dict[str, Union[pd.DataFrame, np.ndarray]],
    target_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Computes AUC for every model on every target and identifies the winning model per target.
    
    Returns:
        DataFrame with rows as targets, columns as model AUCs, plus 'Best Model' and 'Spread'.
    """
    targets = target_names or TARGET_COLS
    if isinstance(y_true, pd.DataFrame):
        y_true_arr = y_true[targets].to_numpy()
    else:
        y_true_arr = np.asarray(y_true)

    model_names = list(preds_dict.keys())
    records = []

    for t_idx, target in enumerate(targets):
        row = {"Target": target}
        aucs = {}
        t_true = y_true_arr[:, t_idx]

        for m_name in model_names:
            p = preds_dict[m_name]
            p_arr = p[targets].to_numpy() if isinstance(p, pd.DataFrame) else np.asarray(p)
            auc = compute_target_auc(t_true, p_arr[:, t_idx])
            row[m_name] = auc
            aucs[m_name] = auc

        best_m = max(aucs, key=aucs.get)
        worst_m = min(aucs, key=aucs.get)
        row["Best Model"] = best_m
        row["AUC Delta (Best - Worst)"] = aucs[best_m] - aucs[worst_m]
        records.append(row)

    df_specialists = pd.DataFrame(records)
    return df_specialists


def format_diversity_summary(
    corr_matrices: Dict[str, pd.DataFrame],
    specialists_df: pd.DataFrame,
) -> str:
    """Creates a comprehensive diagnostic summary in markdown."""
    lines = [
        "### Ensemble Diversity & Specialist Analysis\n",
        "#### Target Specialist Breakdown",
        specialists_df.to_markdown(index=False),
        "\n#### Mean Inter-Model Rank Correlation Across Targets",
    ]
    mean_corrs = []
    for target, mat in corr_matrices.items():
        # Off-diagonal values
        n = mat.shape[0]
        if n > 1:
            off_diag = mat.values[np.triu_indices(n, k=1)]
            mean_corr = float(np.mean(off_diag))
        else:
            mean_corr = 1.0
        mean_corrs.append({"Target": target, "Mean Pairwise Spearman Rho": mean_corr})

    df_corr = pd.DataFrame(mean_corrs)
    lines.append(df_corr.to_markdown(index=False))
    return "\n".join(lines)
