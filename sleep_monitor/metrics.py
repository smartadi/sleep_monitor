"""
Accuracy metrics: compare CAP-derived rates against PSG ground truth.
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import pearsonr


def accuracy_metrics(
    gt_t: np.ndarray,
    gt_hz: np.ndarray,
    cap_t: np.ndarray,
    cap_hz: np.ndarray,
) -> dict:
    """
    Compare a CAP rate estimate against PSG ground truth.

    Both time axes are in the same unit (seconds or hours — must match).
    Ground truth is linearly interpolated onto the CAP time grid.

    Parameters
    ----------
    gt_t   : time axis for ground truth
    gt_hz  : ground truth rate in Hz
    cap_t  : time axis for CAP estimate
    cap_hz : CAP rate estimate in Hz

    Returns
    -------
    dict with keys: n, mae, rmse, r, bias  (all NaN if insufficient data)
    """
    valid_gt  = ~np.isnan(gt_hz)
    valid_cap = ~np.isnan(cap_hz)
    if valid_gt.sum() < 2 or valid_cap.sum() < 2:
        return dict(n=0, mae=np.nan, rmse=np.nan, r=np.nan, bias=np.nan)

    f_gt = interp1d(gt_t[valid_gt], gt_hz[valid_gt],
                    kind='linear', bounds_error=False, fill_value=np.nan)
    t_lo = max(gt_t[valid_gt][0],  cap_t[valid_cap][0])
    t_hi = min(gt_t[valid_gt][-1], cap_t[valid_cap][-1])
    mask = valid_cap & (cap_t >= t_lo) & (cap_t <= t_hi)
    if mask.sum() < 5:
        return dict(n=0, mae=np.nan, rmse=np.nan, r=np.nan, bias=np.nan)

    ref  = f_gt(cap_t[mask])
    pred = cap_hz[mask]
    ok   = ~np.isnan(ref) & ~np.isnan(pred)
    ref, pred = ref[ok], pred[ok]
    if len(ref) < 5:
        return dict(n=0, mae=np.nan, rmse=np.nan, r=np.nan, bias=np.nan)

    err  = pred - ref
    r, _ = pearsonr(ref, pred)
    return dict(
        n    = int(len(ref)),
        mae  = float(np.mean(np.abs(err))),
        rmse = float(np.sqrt(np.mean(err**2))),
        r    = float(r),
        bias = float(np.mean(err)),
    )


def metrics_table(results: Dict[str, Dict[str, dict]]) -> 'pd.DataFrame':
    """
    Flatten a nested results dict into a tidy DataFrame.

    Parameters
    ----------
    results : {session_label: {method: metrics_dict}}
        e.g. {'S1N1': {'acf': {'mae': 0.05, ...}, 'peaks': {...}}, ...}

    Returns
    -------
    pd.DataFrame with columns: session, method, n, mae, rmse, r, bias
    """
    import pandas as pd
    rows = []
    for sess_label, methods in results.items():
        for method, m in methods.items():
            rows.append({'session': sess_label, 'method': method, **m})
    return pd.DataFrame(rows)


def summary_by_method(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """
    Aggregate metrics across sessions, grouped by method.

    Parameters
    ----------
    df : output of metrics_table()

    Returns
    -------
    pd.DataFrame with mean ± std for MAE, RMSE, r, bias per method
    """
    import pandas as pd
    agg = df.groupby('method')[['mae', 'rmse', 'r', 'bias']].agg(['mean', 'std'])
    agg.columns = ['_'.join(c) for c in agg.columns]
    return agg.reset_index()
