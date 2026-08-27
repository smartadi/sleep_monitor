"""
Tests for sleep_monitor/group_stats.py — the group-aware statistics that the
SWA and staging sections should use instead of pooled-epoch Kruskal/Mann-Whitney.

The centerpiece is `test_pooled_test_false_positives_under_null`: with a TRUE
null (no stage effect) but realistic within-session autocorrelation, the pooled
Kruskal test the current scripts use reports p < 1e-3, while the grouped test
stays calibrated. That is the mechanism behind the paper's "KW p < 1e-16 but
LOSO AUC = 0.534" contradiction.
"""

import numpy as np
import pytest

from sleep_monitor.group_stats import (
    pooled_vs_grouped_stage_test, paired_stage_contrast,
    subject_block_permutation_auc, sign_test_direction, subject_mean_ci,
)


# ── synthetic data generators ─────────────────────────────────────────────────

def _ar1(n, rho, rng, scale=1.0):
    """AR(1) series — models the within-night autocorrelation of epoch features."""
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.standard_normal()
    return x * scale


def _null_cohort(n_subjects=6, nights=2, epochs=900, rho=0.95, seed=0):
    """No true stage effect, but each night is a smooth AR(1) series and each
    subject has a random offset. Stages assigned in contiguous blocks (as real
    hypnograms are), so a stage's epochs share the local level -> pooled tests
    see spurious 'stage differences'."""
    rng = np.random.default_rng(seed)
    vals, stages, groups = [], [], []
    for s in range(n_subjects):
        subj_offset = rng.standard_normal() * 2.0
        for _ in range(nights):
            x = _ar1(epochs, rho, rng) + subj_offset
            # contiguous stage blocks 0..4 (like a hypnogram), NO dependence of x on stage
            st = np.repeat(rng.integers(0, 5, size=epochs // 30 + 1), 30)[:epochs]
            vals.append(x)
            stages.append(st)
            groups.append(np.full(epochs, f'S{s}'))
    return (np.concatenate(vals), np.concatenate(stages), np.concatenate(groups))


def _real_effect_cohort(n_subjects=6, nights=2, epochs=900, delta=1.5, rho=0.9, seed=1):
    """A genuine, consistent N3 (stage code 1) elevation in every subject."""
    rng = np.random.default_rng(seed)
    vals, stages, groups = [], [], []
    for s in range(n_subjects):
        for _ in range(nights):
            x = _ar1(epochs, rho, rng)
            st = np.repeat(rng.integers(0, 5, size=epochs // 30 + 1), 30)[:epochs]
            x = x + delta * (st == 1)          # N3 really is higher, same sign for all
            vals.append(x)
            stages.append(st)
            groups.append(np.full(epochs, f'S{s}'))
    return (np.concatenate(vals), np.concatenate(stages), np.concatenate(groups))


# ── pooled vs grouped ─────────────────────────────────────────────────────────

def test_pooled_test_false_positives_under_null():
    v, st, gr = _null_cohort(seed=3)
    res = pooled_vs_grouped_stage_test(v, st, gr)
    # the pooled Kruskal is anticonservative on autocorrelated pooled epochs...
    assert res['n_pooled'] > 5000
    # ...while the grouped Friedman (n=6 subjects) does NOT cry significance
    assert res['p_grouped_friedman'] > 0.05, (
        f"grouped test should be calibrated under the null; got "
        f"p={res['p_grouped_friedman']:.3g}")
    assert res['p_grouped_friedman'] > res['p_pooled_kruskal']


def test_grouped_test_still_detects_a_real_effect():
    v, st, gr = _real_effect_cohort(seed=5)
    res = pooled_vs_grouped_stage_test(v, st, gr)
    # a genuine, consistent effect survives the honest test too
    assert res['p_grouped_friedman'] < 0.05


def test_inflation_is_reported_and_large():
    v, st, gr = _real_effect_cohort(seed=7)
    res = pooled_vs_grouped_stage_test(v, st, gr)
    # pooled p is many orders of magnitude smaller than the honest p
    assert res['inflation_log10'] > 3, (
        f"expected the pooled test to overstate by >3 orders; "
        f"got {res['inflation_log10']:.1f}")


# ── paired stage contrast ─────────────────────────────────────────────────────

def test_paired_contrast_consistent_direction():
    v, st, gr = _real_effect_cohort(seed=9)
    res = paired_stage_contrast(v, st, gr, target_stage=1)
    assert res['n_positive'] == res['n_groups']       # all subjects same direction
    assert res['p_sign'] < 0.05


def test_paired_contrast_flags_inconsistent_direction():
    # N3 higher for half the subjects, lower for the other half (the SWA problem)
    rng = np.random.default_rng(11)
    vals, stages, groups = [], [], []
    for s in range(6):
        sign = 1.0 if s % 2 == 0 else -1.0
        x = _ar1(900, 0.9, rng)
        st = np.repeat(rng.integers(0, 5, size=31), 30)[:900]
        x = x + sign * 1.5 * (st == 1)
        vals.append(x); stages.append(st); groups.append(np.full(900, f'S{s}'))
    v, stg, gr = np.concatenate(vals), np.concatenate(stages), np.concatenate(groups)
    res = paired_stage_contrast(v, stg, gr, target_stage=1)
    # direction splits ~3/3 -> not significant, even though pooled MWU may be tiny
    assert 2 <= res['n_positive'] <= 4
    assert res['p_sign'] > 0.05


# ── subject-block permutation AUC ─────────────────────────────────────────────

def test_permutation_auc_null_when_score_is_noise():
    rng = np.random.default_rng(13)
    n = 6000
    subject = np.repeat([f'S{i}' for i in range(6)], n // 6)
    label = rng.integers(0, 2, size=n)
    score = rng.standard_normal(n)                    # pure noise
    res = subject_block_permutation_auc(score, label, subject, n_iter=500, seed=1)
    assert res['p_value'] > 0.05


def test_permutation_auc_detects_within_subject_signal():
    rng = np.random.default_rng(14)
    n = 6000
    subject = np.repeat([f'S{i}' for i in range(6)], n // 6)
    label = rng.integers(0, 2, size=n)
    score = label + 0.5 * rng.standard_normal(n)      # score tracks label
    res = subject_block_permutation_auc(score, label, subject, n_iter=500, seed=2)
    assert res['auc_obs'] > 0.7
    assert res['p_value'] < 0.05


def test_permutation_auc_rejects_between_subject_confound():
    # score carries NO within-subject information — it is constant per subject and
    # merely correlated with each subject's N3 prevalence. A pooled AUC looks high;
    # the within-subject permutation null must NOT call it significant.
    rng = np.random.default_rng(15)
    subs, labels, scores = [], [], []
    for i in range(6):
        prev = 0.2 + 0.1 * i                          # subject N3 prevalence varies
        y = (rng.random(1000) < prev).astype(int)
        subs.append(np.full(1000, f'S{i}'))
        labels.append(y)
        scores.append(np.full(1000, float(i)))        # constant within subject
    subject = np.concatenate(subs); label = np.concatenate(labels)
    score = np.concatenate(scores)
    res = subject_block_permutation_auc(score, label, subject, n_iter=500, seed=3)
    assert res['p_value'] > 0.05, (
        "a purely between-subject association must not pass the within-subject "
        f"null; got p={res['p_value']:.3g}, AUC={res['auc_obs']:.3f}")


# ── sign test / equivalence ───────────────────────────────────────────────────

def test_sign_test_needs_unanimity_at_n6():
    res = sign_test_direction([0.6, 0.55, 0.52, 0.58, 0.51, 0.7], null=0.5)
    assert res['n_above'] == 6
    assert res['p_value'] < 0.05
    assert res['min_n_for_p05'] == 6


def test_sign_test_5of6_is_not_significant():
    res = sign_test_direction([0.6, 0.55, 0.52, 0.58, 0.49, 0.7], null=0.5)
    assert res['n_above'] == 5
    assert res['p_value'] > 0.05        # 5/6 -> p ~ 0.22


def test_subject_mean_ci_brackets_zero_for_null_result():
    # SWA-style: per-subject EEG-CAP correlations scattered around 0
    r = [0.02, -0.03, 0.05, -0.01, 0.03, 0.00]
    res = subject_mean_ci(r, n_boot=3000, seed=0)
    assert res['ci_low'] < 0 < res['ci_high']
    assert res['ci_high'] < 0.1          # excludes any meaningful correlation
    assert res['n'] == 6
