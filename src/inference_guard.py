"""
RSNA Knee Abnormality Detection: Inference Guard & Integrity Verification

Critical Kaggle Infrastructure Hardening:
1. GPU hardware & torch AMP dtype compatibility check (prevents silent BF16 emulation slowdown on T4)
2. Per-study inference timer with 9-hour runtime projection
3. Zero-compromise submission format and completeness validation
4. Neutral-fill detection & threshold guard
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


from .metrics import TARGET_COLS

EXPECTED_HEADER: List[str] = ["StudyInstanceUID"] + TARGET_COLS
MAX_RUN_TIME_SECONDS: float = 9.0 * 3600.0  # 9 hours Kaggle kernel limit


def verify_gpu_and_dtype() -> Dict[str, Union[str, bool, Tuple[object, bool]]]:
    """
    Validates CUDA device and ensures supported AMP dtype.
    CRITICAL: NVIDIA T4 (Compute Capability 7.5) does NOT have native BF16 hardware units!
    Running BF16 on T4 triggers software emulation, causing 3x-5x slowdown and 9h timeouts.
    """
    if not TORCH_AVAILABLE:
        return {
            "cuda_available": False,
            "device_name": "Torch Not Installed (Local Host)",
            "recommended_dtype": ("float32", False),
            "warning": "PyTorch is not installed in the local environment.",
        }

    status: Dict[str, Union[str, bool, Tuple[object, bool]]] = {
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        status["device_name"] = "CPU"
        status["recommended_dtype"] = (torch.float32, False)
        status["warning"] = "Running on CPU; execution will be slow."
        return status


    dev = torch.cuda.current_device()
    dev_name = torch.cuda.get_device_name(dev)
    cap = torch.cuda.get_device_capability(dev)  # (major, minor)
    status["device_name"] = dev_name
    status["compute_capability"] = f"{cap[0]}.{cap[1]}"

    # Ampere (sm_80, e.g. A100, RTX 3090) or Hopper (sm_90) natively supports BF16
    # Turing (sm_75, e.g. T4) and Pascal (sm_60, e.g. P100) MUST use FP16, NEVER BF16!
    if cap[0] >= 8:
        # Native BF16 supported
        status["recommended_dtype"] = (torch.bfloat16, True)
        status["bf16_native"] = True
    else:
        # T4 or P100: use FP16
        status["recommended_dtype"] = (torch.float16, True)
        status["bf16_native"] = False
        if "t4" in dev_name.lower() or "p100" in dev_name.lower():
            status["dtype_safety_note"] = (
                f"Detected {dev_name} (sm_{cap[0]}{cap[1]}). "
                "FP16 selected. BF16 strictly prohibited to prevent emulation timeout."
            )

    return status


class RuntimeWatchdog:
    """
    Profiles inference speed per study and estimates full test set runtime.
    Raises alerts if projected runtime exceeds Kaggle 9h limit.
    """

    def __init__(
        self,
        total_test_studies: int,
        safety_margin_hours: float = 1.0,
        warmup_studies: int = 3,
    ):
        self.total_test_studies = total_test_studies
        self.max_allowed_seconds = MAX_RUN_TIME_SECONDS - (safety_margin_hours * 3600.0)
        self.warmup_studies = warmup_studies
        self.study_times: List[float] = []
        self.start_wall_time = time.time()

    def record_study(self, study_duration_sec: float) -> None:
        self.study_times.append(study_duration_sec)

    def get_projection(self) -> Dict[str, Union[float, str, bool]]:
        n_done = len(self.study_times)
        if n_done == 0:
            return {"status": "No studies recorded"}

        # Exclude warmup for steady-state estimation
        active_times = self.study_times[self.warmup_studies:] if n_done > self.warmup_studies else self.study_times
        mean_time_per_study = float(np.mean(active_times))
        std_time_per_study = float(np.std(active_times)) if len(active_times) > 1 else 0.0

        projected_total_sec = mean_time_per_study * self.total_test_studies
        projected_total_hours = projected_total_sec / 3600.0
        is_safe = projected_total_sec <= self.max_allowed_seconds

        elapsed = time.time() - self.start_wall_time
        return {
            "completed_studies": n_done,
            "total_studies": self.total_test_studies,
            "mean_sec_per_study": mean_time_per_study,
            "std_sec_per_study": std_time_per_study,
            "elapsed_minutes": elapsed / 60.0,
            "projected_total_hours": projected_total_hours,
            "within_9h_limit": is_safe,
            "margin_remaining_hours": (self.max_allowed_seconds - projected_total_sec) / 3600.0,
        }


def validate_submission_file(
    sub_path: Union[str, Path],
    expected_sample_sub_path: Optional[Union[str, Path]] = None,
    allow_neutral_fill: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Rigorously verifies the final submission.csv:
    1. File exists and is non-empty
    2. Column header matches EXACTLY
    3. Row count and StudyInstanceUIDs match test set
    4. No NaNs, nulls, or infinities
    5. Value range is within [0.0, 1.0]
    6. Identifies if silent neutral fills (0.5) occurred
    """
    errors: List[str] = []
    sub_path = Path(sub_path)

    if not sub_path.is_file():
        return False, [f"File not found: {sub_path}"]

    try:
        df = pd.read_csv(sub_path, dtype={"StudyInstanceUID": str})
    except Exception as e:
        return False, [f"Failed to parse CSV: {e}"]

    # 1. Check header
    if list(df.columns) != EXPECTED_HEADER:
        errors.append(f"Header mismatch. Expected {EXPECTED_HEADER}, got {list(df.columns)}")

    # 2. Check shape & NaN
    if df.empty:
        errors.append("Submission DataFrame is empty.")

    nan_count = int(df[TARGET_COLS].isna().sum().sum())
    if nan_count > 0:
        errors.append(f"Submission contains {nan_count} NaN values!")

    # 3. Check values in range
    numeric_vals = df[TARGET_COLS].to_numpy()
    if np.isneginf(numeric_vals).any() or np.isposinf(numeric_vals).any():
        errors.append("Submission contains infinite values (-inf or inf)!")

    if (numeric_vals < 0.0).any() or (numeric_vals > 1.0).any():
        errors.append("Submission contains confidence scores outside [0.0, 1.0]!")

    # 4. Compare with sample_submission.csv if provided
    if expected_sample_sub_path is not None and Path(expected_sample_sub_path).is_file():
        ref_df = pd.read_csv(expected_sample_sub_path, dtype={"StudyInstanceUID": str})
        if len(df) != len(ref_df):
            errors.append(f"Row count mismatch: got {len(df)}, expected {len(ref_df)}")
        if not (df["StudyInstanceUID"].values == ref_df["StudyInstanceUID"].values).all():
            errors.append("StudyInstanceUID ordering or IDs mismatch with sample_submission.csv")

    # 5. Check neutral 0.5 fills
    neutral_mask = (numeric_vals == 0.5)
    neutral_count = int(np.sum(neutral_mask))
    total_cells = numeric_vals.size
    neutral_ratio = neutral_count / total_cells if total_cells > 0 else 0.0

    if neutral_ratio > 0.05 and not allow_neutral_fill:
        errors.append(
            f"SUSPICIOUS: {neutral_ratio:.1%} of prediction cells are neutral 0.5. "
            "Possible unhandled exception or silent fallback triggered during inference!"
        )

    is_valid = len(errors) == 0
    return is_valid, errors
