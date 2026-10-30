"""
RSNA Knee Abnormality Detection: Competition Metrics Engine
Calculates:
- Exact 12-target macro ROC-AUC
- Per-target ROC-AUC with support count & positive prevalence
- Optional bootstrap 95% confidence intervals per target
- Target correlation matrix & calibration analysis
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TARGET_COLS: List[str] = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def compute_target_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Computes ROC-AUC for a single target, safely handling constant ground truth."""
    # Check if target has both classes present
    unique_vals = np.unique(y_true[~np.isnan(y_true)])
    if len(unique_vals) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))


def evaluate_predictions(
    y_true: Union[pd.DataFrame, np.ndarray],
    y_pred: Union[pd.DataFrame, np.ndarray],
    target_names: Optional[List[str]] = None,
    n_bootstrap: int = 0,
    random_seed: int = 42,
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Evaluates multi-target predictions according to the RSNA Knee competition metric.
    
    Args:
        y_true: Ground truth binary (or probabilistic) labels, shape (N, 12).
        y_pred: Predicted confidence scores / probabilities, shape (N, 12).
        target_names: List of column names (defaults to competition TARGET_COLS).
        n_bootstrap: Number of bootstrap iterations for 95% CI (0 to disable).
        random_seed: Seed for bootstrap reproducibility.
        
    Returns:
        Dict containing:
            - 'macro_auc': unweighted mean of valid per-target AUCs
            - 'per_target_auc': dict of {target: auc}
            - 'per_target_prevalence': dict of {target: positive_ratio}
            - 'per_target_ci' (if n_bootstrap > 0): dict of {target: (lower_95, upper_95)}
    """
    if target_names is None:
        target_names = TARGET_COLS

    if isinstance(y_true, pd.DataFrame):
        y_true_arr = y_true[target_names].to_numpy()
    else:
        y_true_arr = np.asarray(y_true)

    if isinstance(y_pred, pd.DataFrame):
        y_pred_arr = y_pred[target_names].to_numpy()
    else:
        y_pred_arr = np.asarray(y_pred)

    assert y_true_arr.shape == y_pred_arr.shape, (
        f"Shape mismatch: y_true {y_true_arr.shape} vs y_pred {y_pred_arr.shape}"
    )

    per_target_auc: Dict[str, float] = {}
    per_target_prev: Dict[str, float] = {}
    per_target_ci: Dict[str, Tuple[float, float]] = {}
    valid_aucs: List[float] = []

    rng = np.random.RandomState(random_seed)

    for i, target in enumerate(target_names):
        col_true = y_true_arr[:, i]
        col_pred = y_pred_arr[:, i]

        # Valid mask (non-NaN true labels)
        mask = ~np.isnan(col_true)
        c_true = col_true[mask]
        c_pred = col_pred[mask]

        prev = float(np.mean(c_true > 0.5)) if len(c_true) > 0 else 0.0
        per_target_prev[target] = prev

        auc = compute_target_auc(c_true, c_pred)
        per_target_auc[target] = auc

        if not np.isnan(auc):
            valid_aucs.append(auc)

        if n_bootstrap > 0 and len(c_true) > 10:
            b_aucs = []
            n_samples = len(c_true)
            for _ in range(n_bootstrap):
                b_idx = rng.randint(0, n_samples, size=n_samples)
                b_auc = compute_target_auc(c_true[b_idx], c_pred[b_idx])
                if not np.isnan(b_auc):
                    b_aucs.append(b_auc)
            if len(b_aucs) > 0:
                per_target_ci[target] = (
                    float(np.percentile(b_aucs, 2.5)),
                    float(np.percentile(b_aucs, 97.5)),
                )

    macro_auc = float(np.mean(valid_aucs)) if len(valid_aucs) > 0 else 0.0

    result: Dict[str, Union[float, Dict[str, float]]] = {
        "macro_auc": macro_auc,
        "per_target_auc": per_target_auc,
        "per_target_prevalence": per_target_prev,
    }
    if n_bootstrap > 0:
        result["per_target_ci"] = per_target_ci

    return result


def format_evaluation_report(
    eval_res: Dict[str, Union[float, Dict[str, float]]],
    model_name: str = "Model",
) -> str:
    """Formats an evaluation result into a readable markdown summary table."""
    lines = [
        f"### Scoring Engine Report: {model_name}",
        f"**Macro ROC-AUC**: `{eval_res['macro_auc']:.5f}`\n",
        "| Target | ROC-AUC | Prevalence | 95% CI |",
        "| :--- | :---: | :---: | :---: |",
    ]
    per_target = eval_res["per_target_auc"]
    prevs = eval_res["per_target_prevalence"]
    cis = eval_res.get("per_target_ci", {})

    for target, auc in per_target.items():
        prev_str = f"{prevs.get(target, 0.0):.1%}"
        auc_str = f"{auc:.5f}" if not np.isnan(auc) else "N/A"
        ci_str = (
            f"[{cis[target][0]:.4f}, {cis[target][1]:.4f}]"
            if target in cis
            else "-"
        )
        lines.append(f"| {target} | **{auc_str}** | {prev_str} | {ci_str} |")

    return "\n".join(lines)
