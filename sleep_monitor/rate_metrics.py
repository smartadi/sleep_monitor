"""
Paper-level evaluation metrics for the CAP rate-estimation claims.

`metrics.py` scores one CAP estimate against ground truth per session (MAE, r,
bias on pooled epochs). That is the *within-night* view. It does **not** answer
the questions a reviewer asks of the manuscript's rate section:

  1. Does the estimator carry information across the frequency band, or is it
     degenerate/quantized?  -> frequency_sweep, degeneracy_report, octave_report
  2. Is the reported night-level agreement a real measurement or a residual of a
     scale factor `k` fitted on the same recording it is scored on?
     -> evaluate_calibration_regimes  (self / cross-night / population / no-sensor)
  3. Does the within-night rate track beat a null that destroys the time
     alignment but keeps the marginal distribution? -> circular_shift_null
  4. Are the small-n (n<=6) correlations significant under an *exact* test rather
     than the invalid t-approximation? -> exact_spearman_perm
  5. Are Bland-Altman limits computed on the correct unit of analysis (the
     recording), not on pooled non-independent epochs? -> bland_altman

Every function here is pure (numpy/scipy/pandas only) and unit-testable without
the overnight recordings. The estimator-family functions take a callable
`estimator(x, f_lo, f_hi, fs) -> rate_Hz`, so they work on any of the
`sleep_monitor.rates` estimators or a custom one.

See `writeup/edits/RATE_AUDIT_2026-08-06.md` and `analysis/rates/CLAUDE.md` for
the findings these metrics operationalize.
"""

from __future__ import annotations

from typing import Callable, Sequence
import itertools

import numpy as np

Estimator = Callable[..., float]


# ── 1. Estimator behaviour across the frequency band ──────────────────────────

def frequency_sweep(
    estimator: Estimator,
    f_lo: float,
    f_hi: float,
    fs: float,
    freqs: Sequence[float] | None = None,
    win_s: float = 60.0,
    amplitude: float = 1.0,
) -> dict:
    """Run an estimator on clean sinusoids spanning the band.

    Returns a dict of parallel arrays::

        {'true_hz', 'est_hz', 'abs_err_hz', 'signed_err_hz'}

    A correct estimator tracks the diagonal (est ~= true) across the whole band.
    The two classic failure modes this exposes:

      * **quantization / degeneracy** — `est_hz` takes only a handful of distinct
        values regardless of `true_hz` (see `degeneracy_report`);
      * **octave / subharmonic locking** — `est_hz ~= true_hz / 2` in part of the
        band (see `octave_report`).

    The sinusoid is the most favourable possible input; anything an estimator
    gets wrong here it gets wrong on real signals too.
    """
    if freqs is None:
        # avoid the band edges; sample densely enough to see quantization
        freqs = np.linspace(f_lo * 1.15, f_hi * 0.95, 40)
    freqs = np.asarray(freqs, dtype=float)
    n = int(round(win_s * fs))
    t = np.arange(n) / fs
    est = np.empty(len(freqs))
    for i, f in enumerate(freqs):
        x = amplitude * np.sin(2 * np.pi * f * t)
        try:
            est[i] = float(estimator(x, f_lo, f_hi, fs))
        except Exception:
            est[i] = np.nan
    return {
        'true_hz': freqs,
        'est_hz': est,
        'abs_err_hz': np.abs(est - freqs),
        'signed_err_hz': est - freqs,
    }


def degeneracy_report(sweep: dict, round_hz: float = 1e-3) -> dict:
    """Summarize how much a `frequency_sweep` collapses distinct inputs.

    Returns::

        {'n_inputs', 'n_distinct_outputs', 'distinct_fraction',
         'is_degenerate', 'modal_output_hz', 'modal_fraction'}

    `is_degenerate` is True when the estimator produces very few distinct
    outputs (distinct_fraction < 0.25) — the signature of `rate_spectral` in the
    respiratory band, which returns 15 br/min almost everywhere.
    """
    est = sweep['est_hz']
    finite = est[np.isfinite(est)]
    if finite.size == 0:
        return dict(n_inputs=len(est), n_distinct_outputs=0,
                    distinct_fraction=0.0, is_degenerate=True,
                    modal_output_hz=np.nan, modal_fraction=np.nan)
    q = np.round(finite / round_hz) * round_hz
    vals, counts = np.unique(q, return_counts=True)
    modal = int(np.argmax(counts))
    frac_distinct = len(vals) / finite.size
    return dict(
        n_inputs=len(est),
        n_distinct_outputs=int(len(vals)),
        distinct_fraction=float(frac_distinct),
        is_degenerate=bool(frac_distinct < 0.25),
        modal_output_hz=float(vals[modal]),
        modal_fraction=float(counts[modal] / finite.size),
    )


def octave_report(sweep: dict, rel_tol: float = 0.06) -> dict:
    """Count band points where the estimate is a half/double of the truth.

    Returns fractions of the swept band where ``est ~= true/2`` (subharmonic
    locking) or ``est ~= 2*true`` (harmonic locking). A nonzero
    ``half_fraction`` means the estimator reports half the real rate somewhere in
    the operating band — a silent, physiologically plausible-looking error.
    """
    true = sweep['true_hz']
    est = sweep['est_hz']
    ok = np.isfinite(est) & (true > 0)
    if ok.sum() == 0:
        return dict(half_fraction=np.nan, double_fraction=np.nan,
                    on_diagonal_fraction=np.nan, half_freqs=[])
    rel = np.abs(est[ok] - true[ok]) / true[ok]
    half = np.abs(est[ok] - true[ok] / 2.0) / true[ok] < rel_tol
    dbl = np.abs(est[ok] - 2.0 * true[ok]) / true[ok] < rel_tol
    diag = rel < rel_tol
    return dict(
        half_fraction=float(half.mean()),
        double_fraction=float(dbl.mean()),
        on_diagonal_fraction=float(diag.mean()),
        half_freqs=[float(f) for f in true[ok][half]],
    )


# ── 2. Calibration honesty: is k a measurement or a same-night fit? ───────────

def fit_k(raw: np.ndarray, ref: np.ndarray, clip=(0.3, 5.0)) -> float:
    """Overcount factor k = median(raw / ref), clipped, as the pipeline defines it.

    Both arrays are per-epoch rates in the same unit; NaNs are ignored. The
    calibrated estimate is then ``raw / k``.
    """
    raw = np.asarray(raw, float)
    ref = np.asarray(ref, float)
    ok = np.isfinite(raw) & np.isfinite(ref) & (ref > 0)
    if ok.sum() < 1:
        return np.nan
    ratios = np.clip(raw[ok] / ref[ok], *clip)
    return float(np.median(ratios))


def night_mean_error(raw: np.ndarray, ref: np.ndarray, k: float) -> float:
    """|mean(raw/k) - mean(ref)| — the true night-level error of a calibrated estimate.

    This is the quantity a "screening-level nightly rate" claim rests on, and it
    is *not* the median per-epoch |error| the manuscript's Table 3 prints.
    """
    if not np.isfinite(k) or k <= 0:
        return np.nan
    raw = np.asarray(raw, float)
    ref = np.asarray(ref, float)
    cal = raw / k
    m_cal = np.nanmean(cal)
    m_ref = np.nanmean(ref)
    if not (np.isfinite(m_cal) and np.isfinite(m_ref)):
        return np.nan
    return float(abs(m_cal - m_ref))


def evaluate_calibration_regimes(nights: list[dict]) -> "pd.DataFrame":
    """Score night-mean rate error under four calibration regimes.

    This is the central defensibility test for the rate section. Each element of
    `nights` is a dict::

        {'subject': str, 'night': str,
         'raw': 1-D array of per-epoch raw estimates,
         'ref': 1-D array of per-epoch reference rates (same length/grid)}

    For every night it computes ``|mean(raw/k) - mean(ref)|`` under:

      * **self**        — k fitted on the same night (what the paper reports; a
                          fitting residual, an upper bound on performance);
      * **cross_night** — k from the same subject's *other* night (realistic
                          deployment: calibrate once, use later);
      * **population**  — one k = median of all other nights' self-k (leave the
                          night out);
      * **no_sensor**   — ignore the sensor entirely and predict the
                          leave-one-subject-out cohort **median reference night
                          mean**. This is the bar any sensor must clear.

    Returns a tidy DataFrame, one row per night, with those four error columns.
    Use `paired_wilcoxon_vs_baseline` on it to test whether any regime beats
    `no_sensor`. On the CAP data, only `self` does — and that is circular.
    """
    import pandas as pd

    # Precompute per-night self-k and reference night means.
    for nd in nights:
        nd['_k_self'] = fit_k(nd['raw'], nd['ref'])
        nd['_ref_mean'] = float(np.nanmean(nd['ref']))

    rows = []
    for i, nd in enumerate(nights):
        subj = nd['subject']
        k_self = nd['_k_self']

        # cross-night: same subject, different night (first match)
        k_cross = np.nan
        for j, other in enumerate(nights):
            if j != i and other['subject'] == subj:
                k_cross = other['_k_self']
                break

        # population k: median self-k of all *other* nights
        others_k = [nights[j]['_k_self'] for j in range(len(nights))
                    if j != i and np.isfinite(nights[j]['_k_self'])]
        k_pop = float(np.median(others_k)) if others_k else np.nan

        # no-sensor: LOSO cohort median reference night-mean
        others_ref = [nights[j]['_ref_mean'] for j in range(len(nights))
                      if nights[j]['subject'] != subj
                      and np.isfinite(nights[j]['_ref_mean'])]
        pred_no_sensor = float(np.median(others_ref)) if others_ref else np.nan
        err_no_sensor = (abs(pred_no_sensor - nd['_ref_mean'])
                         if np.isfinite(pred_no_sensor) else np.nan)

        rows.append({
            'subject': subj,
            'night': nd['night'],
            'k_self': k_self,
            'ref_mean': nd['_ref_mean'],
            'err_self': night_mean_error(nd['raw'], nd['ref'], k_self),
            'err_cross_night': night_mean_error(nd['raw'], nd['ref'], k_cross),
            'err_population': night_mean_error(nd['raw'], nd['ref'], k_pop),
            'err_no_sensor': err_no_sensor,
        })
    return pd.DataFrame(rows)


def paired_wilcoxon_vs_baseline(
    df: "pd.DataFrame",
    regime_col: str,
    baseline_col: str = 'err_no_sensor',
) -> dict:
    """Paired Wilcoxon: is `regime_col` error smaller than `baseline_col`?

    Returns ``{'n', 'median_regime', 'median_baseline', 'statistic', 'p_value',
    'beats_baseline'}``. `beats_baseline` is True only when the regime's median
    error is smaller *and* p < 0.05. The recording is the unit of analysis, so n
    is the number of nights, not the number of epochs.
    """
    from scipy.stats import wilcoxon

    a = df[regime_col].to_numpy(float)
    b = df[baseline_col].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    med_a, med_b = float(np.median(a)), float(np.median(b))
    if len(a) < 3 or np.allclose(a, b):
        return dict(n=int(len(a)), median_regime=med_a, median_baseline=med_b,
                    statistic=np.nan, p_value=np.nan,
                    beats_baseline=False)
    stat, p = wilcoxon(a, b)
    return dict(n=int(len(a)), median_regime=med_a, median_baseline=med_b,
                statistic=float(stat), p_value=float(p),
                beats_baseline=bool(med_a < med_b and p < 0.05))


# ── 3. Within-night tracking against a circular-shift null ────────────────────

def circular_shift_null(
    cap: np.ndarray,
    ref: np.ndarray,
    n_iter: int = 2000,
    seed: int = 0,
    min_shift_frac: float = 0.05,
) -> dict:
    """Test whether the within-night cap-vs-ref correlation beats chance.

    The null keeps each series' marginal distribution and autocorrelation but
    destroys the time alignment by circularly shifting `cap`. This is the right
    null for "the estimate tracks the reference *over the night*": a smooth
    estimate and a smooth reference correlate spuriously under a naive test, and
    the circular shift removes exactly that.

    Returns ``{'r_obs', 'p_value', 'null_mean', 'null_hi95', 'n'}``.
    `p_value` is the one-sided fraction of shifts with r >= r_obs.
    """
    from scipy.stats import pearsonr

    cap = np.asarray(cap, float)
    ref = np.asarray(ref, float)
    ok = np.isfinite(cap) & np.isfinite(ref)
    cap, ref = cap[ok], ref[ok]
    n = len(cap)
    if n < 10 or np.std(cap) == 0 or np.std(ref) == 0:
        return dict(r_obs=np.nan, p_value=np.nan, null_mean=np.nan,
                    null_hi95=np.nan, n=int(n))
    r_obs = float(pearsonr(cap, ref)[0])
    rng = np.random.default_rng(seed)
    lo = max(1, int(min_shift_frac * n))
    shifts = rng.integers(lo, n - lo, size=n_iter)
    ref_c = ref - ref.mean()
    cap_c = cap - cap.mean()
    denom = np.sqrt(np.sum(cap_c**2) * np.sum(ref_c**2))
    null = np.empty(n_iter)
    for i, s in enumerate(shifts):
        null[i] = np.sum(np.roll(cap_c, s) * ref_c) / denom
    p = float((np.sum(null >= r_obs) + 1) / (n_iter + 1))
    return dict(r_obs=r_obs, p_value=p, null_mean=float(null.mean()),
                null_hi95=float(np.percentile(null, 95)), n=int(n))


# ── 4. Exact small-n statistics ───────────────────────────────────────────────

def exact_spearman_perm(x: np.ndarray, y: np.ndarray) -> dict:
    """Spearman rho with an *exact* two-sided permutation p-value.

    For n <= 6 the t-approximation scipy uses by default is invalid (the
    manuscript's k-vs-age p = 0.042 came from it; the exact value is ~0.058).
    This enumerates all n! label permutations when n <= 8, else samples.

    Returns ``{'rho', 'p_exact', 'n', 'method', 'min_abs_rho_for_p05'}``.
    `min_abs_rho_for_p05` is the smallest |rho| that could reach p < 0.05 at this
    n — at n = 6 it is 0.829, so a single ordering flip destroys significance.
    """
    from scipy.stats import rankdata

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3:
        return dict(rho=np.nan, p_exact=np.nan, n=int(n), method='n<3',
                    min_abs_rho_for_p05=np.nan)
    rx, ry = rankdata(x), rankdata(y)

    def _rho(a, b):
        a = a - a.mean()
        b = b - b.mean()
        d = np.sqrt(np.sum(a**2) * np.sum(b**2))
        return np.sum(a * b) / d if d > 0 else 0.0

    rho_obs = _rho(rx, ry)

    if n <= 8:
        perms = itertools.permutations(range(n))
        count = 0
        total = 0
        for p in perms:
            total += 1
            if abs(_rho(rx, ry[list(p)])) >= abs(rho_obs) - 1e-12:
                count += 1
        p_exact = count / total
        method = 'exact'
    else:
        rng = np.random.default_rng(0)
        n_iter = 20000
        count = 1
        for _ in range(n_iter):
            if abs(_rho(rx, rng.permutation(ry))) >= abs(rho_obs) - 1e-12:
                count += 1
        p_exact = count / (n_iter + 1)
        method = 'sampled'

    # smallest |rho| whose exact two-sided p can dip below 0.05 at this n
    thresh = _min_abs_rho_for_p05(n) if n <= 8 else np.nan
    return dict(rho=float(rho_obs), p_exact=float(p_exact), n=int(n),
                method=method, min_abs_rho_for_p05=thresh)


def _min_abs_rho_for_p05(n: int) -> float:
    """Smallest |Spearman rho| reaching two-sided exact p < 0.05 for this n."""
    from scipy.stats import rankdata
    base = np.arange(n)
    rho_vals = []
    for p in itertools.permutations(range(n)):
        a = base - base.mean()
        b = np.array(p, float) - np.mean(p)
        d = np.sqrt(np.sum(a**2) * np.sum(b**2))
        rho_vals.append(abs(np.sum(a * b) / d) if d > 0 else 0.0)
    rho_vals = np.array(rho_vals)
    total = len(rho_vals)
    # find the smallest observed |rho| whose tail probability is < 0.05
    for r in np.unique(rho_vals):
        p = np.mean(rho_vals >= r - 1e-12)
        if p < 0.05:
            return float(r)
    return float('nan')


# ── 5. Bland-Altman on the correct unit of analysis ───────────────────────────

def bland_altman(cap: np.ndarray, ref: np.ndarray) -> dict:
    """Bias and 95% limits of agreement between paired measurements.

    Returns ``{'n', 'bias', 'sd', 'loa_low', 'loa_high'}``.

    **Feed this one value per recording, not one per epoch.** Pooling epochs from
    12 nights inflates n from 12 to ~90,000 and reports limits far tighter than
    the night-to-night agreement a deployment would see; the epochs within a
    night are not independent. `n` in the result is your reminder — if it is in
    the thousands you are computing the wrong quantity.
    """
    cap = np.asarray(cap, float)
    ref = np.asarray(ref, float)
    ok = np.isfinite(cap) & np.isfinite(ref)
    cap, ref = cap[ok], ref[ok]
    n = len(cap)
    if n < 2:
        return dict(n=int(n), bias=np.nan, sd=np.nan,
                    loa_low=np.nan, loa_high=np.nan)
    diff = cap - ref
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    return dict(n=int(n), bias=bias, sd=sd,
                loa_low=bias - 1.96 * sd, loa_high=bias + 1.96 * sd)
