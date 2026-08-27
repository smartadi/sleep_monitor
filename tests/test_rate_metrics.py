"""
Tests for sleep_monitor/rate_metrics.py — the paper-level evaluation metrics.

Each metric is validated against a synthetic case with a known answer, so the
metric itself is trustworthy before it is turned on the overnight recordings.
"""

import numpy as np
import pytest

from sleep_monitor.rates import rate_spectral, rate_spectral_interp, rate_acf
from sleep_monitor.rate_metrics import (
    frequency_sweep, degeneracy_report, octave_report,
    fit_k, night_mean_error, evaluate_calibration_regimes,
    paired_wilcoxon_vs_baseline, circular_shift_null,
    exact_spearman_perm, bland_altman,
)

FS = 100.0
RESP_LO, RESP_HI = 0.1, 0.5


# ── frequency sweep / degeneracy / octave ─────────────────────────────────────

def test_frequency_sweep_shapes():
    sw = frequency_sweep(rate_spectral_interp, RESP_LO, RESP_HI, FS,
                         freqs=np.linspace(0.12, 0.48, 10))
    for key in ('true_hz', 'est_hz', 'abs_err_hz', 'signed_err_hz'):
        assert len(sw[key]) == 10


def test_degeneracy_report_catches_constant():
    sw = frequency_sweep(rate_spectral, RESP_LO, RESP_HI, FS)
    assert degeneracy_report(sw)['is_degenerate'] is True


def test_degeneracy_report_passes_good_estimator():
    sw = frequency_sweep(rate_spectral_interp, RESP_LO, RESP_HI, FS)
    assert degeneracy_report(sw)['is_degenerate'] is False


def test_octave_report_detects_half_locking():
    sw = frequency_sweep(rate_acf, RESP_LO, RESP_HI, FS,
                         freqs=np.linspace(0.12, 0.48, 40))
    assert octave_report(sw)['half_fraction'] > 0.2


# ── fit_k / night_mean_error ──────────────────────────────────────────────────

def test_fit_k_recovers_overcount():
    ref = np.full(200, 0.20)
    raw = ref * 1.5              # estimator overcounts by 1.5x
    assert fit_k(raw, ref) == pytest.approx(1.5, abs=1e-6)


def test_fit_k_clips_extremes():
    ref = np.array([0.2, 0.2, 0.2, 1e-6])   # one near-zero ref -> huge ratio, clipped
    raw = np.array([0.3, 0.3, 0.3, 0.3])
    assert fit_k(raw, ref) <= 5.0


def test_night_mean_error_zero_when_calibrated_perfectly():
    ref = 0.20 + 0.01 * np.sin(np.arange(300))
    raw = ref * 1.5
    k = fit_k(raw, ref)
    assert night_mean_error(raw, ref, k) == pytest.approx(0.0, abs=1e-3)


# ── evaluate_calibration_regimes ──────────────────────────────────────────────

def _make_nights(seed=0):
    """6 subjects x 2 nights. raw = ref * subject_k + noise; ref mean varies by subject.

    Construction guarantees the ground truth the metric must recover:
      * self-k calibration -> ~0 night-mean error (it's a same-night fit);
      * cross-night-k -> small error (subject_k is stable, ref mean drifts a bit);
      * no-sensor baseline -> the spread of subject ref-means.
    """
    rng = np.random.default_rng(seed)
    subj_ref = {f'S{i}': 0.18 + 0.01 * i for i in range(1, 7)}  # 0.19..0.24 Hz
    subj_k = {f'S{i}': 1.4 + 0.05 * i for i in range(1, 7)}
    nights = []
    for s in subj_ref:
        for ngt in ('N1', 'N2'):
            ref = subj_ref[s] + 0.002 * rng.standard_normal(400)
            raw = ref * subj_k[s] + 0.001 * rng.standard_normal(400)
            nights.append(dict(subject=s, night=ngt, raw=raw, ref=ref))
    return nights


def test_self_calibration_is_near_zero_error():
    df = evaluate_calibration_regimes(_make_nights())
    # same-night k is a fit; its night-mean error must be tiny
    assert df['err_self'].median() < 0.005


def test_self_beats_no_sensor_but_it_is_circular():
    df = evaluate_calibration_regimes(_make_nights())
    res = paired_wilcoxon_vs_baseline(df, 'err_self')
    assert res['median_regime'] < res['median_baseline']
    # this is exactly the circular result the audit warns about: self-k "wins"


def test_no_sensor_column_is_positive_and_finite():
    df = evaluate_calibration_regimes(_make_nights())
    assert np.all(np.isfinite(df['err_no_sensor']))
    assert df['err_no_sensor'].median() > 0


def test_cross_night_worse_than_self():
    df = evaluate_calibration_regimes(_make_nights())
    assert df['err_cross_night'].median() >= df['err_self'].median()


# ── circular_shift_null ───────────────────────────────────────────────────────

def test_circular_null_rejects_real_tracking():
    # cap really tracks ref -> should beat the null (small p)
    t = np.linspace(0, 1, 500)
    ref = np.sin(2 * np.pi * 2 * t) + 0.1 * np.random.default_rng(1).standard_normal(500)
    cap = ref + 0.1 * np.random.default_rng(2).standard_normal(500)
    res = circular_shift_null(cap, ref, n_iter=1000, seed=3)
    assert res['r_obs'] > 0.8
    assert res['p_value'] < 0.05


def test_circular_null_passes_two_independent_smooth_series():
    # two smooth but unrelated series: naive pearson may look nonzero, null should not reject
    rng = np.random.default_rng(4)
    smooth = lambda: np.convolve(rng.standard_normal(2000), np.ones(200) / 200, 'same')
    reject = 0
    for _ in range(20):
        if circular_shift_null(smooth(), smooth(), n_iter=500)['p_value'] < 0.05:
            reject += 1
    # false-positive rate should be near alpha, nowhere near a naive-test rate
    assert reject <= 4


# ── exact_spearman_perm ───────────────────────────────────────────────────────

def test_exact_perm_perfect_monotonic():
    x = np.arange(6.0)
    y = np.arange(6.0)
    res = exact_spearman_perm(x, y)
    assert res['rho'] == pytest.approx(1.0)
    # perfect rank order at n=6: exact two-sided p = 2/720
    assert res['p_exact'] == pytest.approx(2 / 720, abs=1e-6)


def test_n6_two_sided_critical_rho_is_0886():
    # Standard exact critical value: at n=6 you need |rho| >= 0.886 for two-sided
    # p < 0.05. (0.829 is the ONE-sided value — a common confusion.)
    res = exact_spearman_perm(np.arange(6.0), np.arange(6.0))
    assert res['min_abs_rho_for_p05'] == pytest.approx(0.886, abs=0.005)


def test_reported_k_vs_age_rho_cannot_reach_p05():
    # The manuscript reports rho = -0.83 at n=6. Since |−0.83| < 0.886 it cannot
    # reach exact two-sided p < 0.05 — the published p = 0.042 came from the
    # invalid t-approximation.
    crit = exact_spearman_perm(np.arange(6.0), np.arange(6.0))['min_abs_rho_for_p05']
    assert 0.83 < crit, "rho=-0.83 sits below the exact significance threshold"


def test_t_approximation_is_anticonservative_at_n6():
    # A single rank inversion (rho = -0.943): the exact permutation p is several
    # times larger than scipy's default t-approximation. This is the mechanism by
    # which the reported p = 0.042 understated the true uncertainty.
    from scipy.stats import spearmanr
    age = np.array([25, 30, 40, 50, 60, 66], float)
    k = np.array([1.04, 1.02, 0.96, 0.98, 0.93, 0.91])   # one local inversion
    res = exact_spearman_perm(age, k)
    _, p_t = spearmanr(age, k)
    assert res['p_exact'] > p_t * 2.0, (
        f"exact p ({res['p_exact']:.4f}) should be well above the t-approx "
        f"({p_t:.4f}) at n=6")


# ── bland_altman ──────────────────────────────────────────────────────────────

def test_bland_altman_basic():
    ref = np.array([60., 62, 58, 61, 59, 63, 60, 57])
    cap = ref + 1.0
    res = bland_altman(cap, ref)
    assert res['bias'] == pytest.approx(1.0)
    assert res['n'] == 8


def test_bland_altman_n_flags_pooled_epochs():
    # feeding thousands of epochs -> n in the thousands is the reviewer's red flag
    rng = np.random.default_rng(0)
    ref = 60 + rng.standard_normal(90000)
    cap = ref + rng.standard_normal(90000)
    res = bland_altman(cap, ref)
    assert res['n'] > 10000   # metric surfaces the wrong-unit mistake via n
