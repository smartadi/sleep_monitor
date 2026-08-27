"""
Group-aware statistics for the per-stage / SWA claims.

The SWA and staging analyses test whether a CAP feature differs by sleep stage,
almost always with `kruskal(*groups)` or `mannwhitneyu(n3, rest)` where the
groups are **epochs pooled across all 12 sessions** (e.g.
`analysis/slow_wave/paper_ridge_demo.py:405`, `run_harmonic_allsessions.py:115`,
`analysis/mean_value/mean_value_vs_stage.py:326`). With ~10^4–10^5 epochs those
tests return p < 1e-16 for effects that are near-chance out of sample — the
slow-wave `CLAUDE.md` records exactly this contradiction: "KW p<1e-16 but LOSO
AUC = 0.534". The p-value is an artifact of treating autocorrelated epochs from
6 subjects as independent observations; the AUC is the honest number.

This module provides the tests those sections should use instead:

  * `pooled_vs_grouped_stage_test` — runs BOTH the naive pooled Kruskal and the
    correct within-subject test (Friedman on per-subject stage medians) and
    reports the p-value inflation so the artifact is visible.
  * `paired_stage_contrast` — the "N3 vs rest" contrast done per subject, then a
    Wilcoxon signed-rank + exact sign test across the 6 subjects.
  * `subject_block_permutation_auc` — an honest null for a per-epoch classifier
    AUC: permute stage labels *within subject blocks*, so the null respects the
    grouping the pooled test ignores.
  * `sign_test_direction` — exact binomial test for "k of n subjects show the
    effect in the same direction" (the subject-dependent-direction problem the
    SWA ridge result has).
  * `subject_mean_ci` — bootstrap CI over subjects, for stating a negative
    result as an equivalence bound ("cohort mean |r| < 0.1") rather than "p>0.05".

Unit of analysis is the **subject** (or session) throughout, never the epoch.
Pure numpy/scipy/pandas; no data dependency.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np


# ── per-subject aggregation ───────────────────────────────────────────────────

def _per_group_stage_median(values, stage, group):
    """Return a {group: {stage: median_value}} nested dict, ignoring NaNs."""
    import pandas as pd
    df = pd.DataFrame({'v': np.asarray(values, float),
                       'stage': np.asarray(stage),
                       'group': np.asarray(group)})
    df = df[np.isfinite(df['v'])]
    out = {}
    for g, gd in df.groupby('group'):
        out[g] = gd.groupby('stage')['v'].median().to_dict()
    return out


def pooled_vs_grouped_stage_test(
    values: Sequence[float],
    stage: Sequence,
    group: Sequence,
    min_epochs_per_cell: int = 10,
) -> dict:
    """Compare the naive pooled Kruskal-Wallis with the correct grouped test.

    `values` — per-epoch feature; `stage` — stage label per epoch; `group` —
    subject (or session) label per epoch.

    Returns::

        {'p_pooled_kruskal', 'n_pooled', 'p_grouped_friedman', 'n_groups',
         'n_stages_used', 'inflation_log10', 'stages_used'}

    `p_pooled_kruskal` is what the current scripts report (epochs as the unit).
    `p_grouped_friedman` treats each subject's per-stage median as one paired
    observation — a repeated-measures design with n = n_groups, the honest unit.
    `inflation_log10` = log10(p_grouped / p_pooled): how many orders of magnitude
    the pooled test overstates the evidence. It is routinely 10+ .
    """
    from scipy.stats import kruskal, friedmanchisquare
    import pandas as pd

    v = np.asarray(values, float)
    st = np.asarray(stage)
    gr = np.asarray(group)
    ok = np.isfinite(v)
    v, st, gr = v[ok], st[ok], gr[ok]

    # pooled Kruskal across stages (the current practice)
    stages_all = [s for s in pd.unique(st)]
    pooled_groups = [v[st == s] for s in stages_all]
    pooled_groups = [g for g in pooled_groups if len(g) >= min_epochs_per_cell]
    if len(pooled_groups) >= 2:
        _, p_pooled = kruskal(*pooled_groups)
        n_pooled = int(sum(len(g) for g in pooled_groups))
    else:
        p_pooled, n_pooled = np.nan, 0

    # grouped: per-subject per-stage median -> Friedman on stages present in ALL groups
    per = _per_group_stage_median(v, st, gr)
    groups = list(per.keys())
    # a stage is usable only if every group has it
    stage_sets = [set(per[g].keys()) for g in groups]
    common_stages = set.intersection(*stage_sets) if stage_sets else set()
    common_stages = sorted(common_stages)
    if len(groups) >= 3 and len(common_stages) >= 3:
        cols = [[per[g][s] for g in groups] for s in common_stages]
        _, p_grouped = friedmanchisquare(*cols)
    else:
        p_grouped = np.nan

    infl = (np.log10(p_grouped) - np.log10(p_pooled)
            if np.isfinite(p_grouped) and np.isfinite(p_pooled)
            and p_pooled > 0 and p_grouped > 0 else np.nan)

    return dict(
        p_pooled_kruskal=float(p_pooled) if np.isfinite(p_pooled) else np.nan,
        n_pooled=n_pooled,
        p_grouped_friedman=float(p_grouped) if np.isfinite(p_grouped) else np.nan,
        n_groups=len(groups),
        n_stages_used=len(common_stages),
        stages_used=common_stages,
        inflation_log10=float(infl) if np.isfinite(infl) else np.nan,
    )


def paired_stage_contrast(
    values: Sequence[float],
    stage: Sequence,
    group: Sequence,
    target_stage,
    min_epochs: int = 10,
) -> dict:
    """"target vs rest" contrast, computed per subject then tested across subjects.

    For each group: median(target_stage) − median(all other stages). Then a
    Wilcoxon signed-rank test and an exact sign test on those per-group effects,
    plus the naive pooled Mann-Whitney p for comparison.

    Returns::

        {'effects', 'groups', 'n_groups', 'n_positive',
         'p_wilcoxon', 'p_sign', 'p_pooled_mwu', 'median_effect'}

    The per-subject effects also expose the direction-consistency problem: if
    `n_positive` is ~half of `n_groups`, the effect flips sign across subjects
    and no pooled test is meaningful.
    """
    from scipy.stats import wilcoxon, mannwhitneyu, binomtest
    import pandas as pd

    v = np.asarray(values, float)
    st = np.asarray(stage)
    gr = np.asarray(group)
    ok = np.isfinite(v)
    v, st, gr = v[ok], st[ok], gr[ok]

    effects, groups = [], []
    for g in pd.unique(gr):
        m = gr == g
        tgt = v[m & (st == target_stage)]
        rest = v[m & (st != target_stage)]
        if len(tgt) >= min_epochs and len(rest) >= min_epochs:
            effects.append(float(np.median(tgt) - np.median(rest)))
            groups.append(g)
    effects = np.array(effects)

    # pooled Mann-Whitney (the current practice)
    tgt_all = v[st == target_stage]
    rest_all = v[st != target_stage]
    if len(tgt_all) >= min_epochs and len(rest_all) >= min_epochs:
        _, p_pooled = mannwhitneyu(tgt_all, rest_all, alternative='two-sided')
    else:
        p_pooled = np.nan

    n = len(effects)
    n_pos = int(np.sum(effects > 0))
    if n >= 3 and not np.allclose(effects, 0):
        try:
            _, p_w = wilcoxon(effects)
        except ValueError:
            p_w = np.nan
    else:
        p_w = np.nan
    p_sign = binomtest(n_pos, n, 0.5).pvalue if n >= 1 else np.nan

    return dict(
        effects=effects.tolist(),
        groups=list(groups),
        n_groups=n,
        n_positive=n_pos,
        p_wilcoxon=float(p_w) if np.isfinite(p_w) else np.nan,
        p_sign=float(p_sign) if np.isfinite(p_sign) else np.nan,
        p_pooled_mwu=float(p_pooled) if np.isfinite(p_pooled) else np.nan,
        median_effect=float(np.median(effects)) if n else np.nan,
    )


# ── honest classifier null: permute labels within subject blocks ──────────────

def subject_block_permutation_auc(
    score: Sequence[float],
    label: Sequence[int],
    subject: Sequence,
    n_iter: int = 2000,
    seed: int = 0,
) -> dict:
    """AUC with a null that permutes labels *within* each subject.

    A pooled AUC over epochs from 6 subjects can look above chance purely because
    the score's subject-level mean happens to track the subject's N3 prevalence.
    Permuting the binary label within each subject block destroys any within-
    subject stage association while preserving each subject's marginal prevalence
    and the score distribution — the correct null for "does the score separate
    stages *within* a night".

    Returns ``{'auc_obs', 'p_value', 'null_mean', 'null_lo95', 'null_hi95', 'n'}``.
    """
    from sklearn.metrics import roc_auc_score

    s = np.asarray(score, float)
    y = np.asarray(label, int)
    subj = np.asarray(subject)
    ok = np.isfinite(s)
    s, y, subj = s[ok], y[ok], subj[ok]
    if len(np.unique(y)) < 2 or len(s) < 20:
        return dict(auc_obs=np.nan, p_value=np.nan, null_mean=np.nan,
                    null_lo95=np.nan, null_hi95=np.nan, n=int(len(s)))
    auc_obs = float(roc_auc_score(y, s))
    # observed distance from chance is what we test, two-sided
    obs_dev = abs(auc_obs - 0.5)

    rng = np.random.default_rng(seed)
    subj_ids = np.unique(subj)
    idx_by_subj = {u: np.where(subj == u)[0] for u in subj_ids}
    null = np.empty(n_iter)
    for i in range(n_iter):
        y_perm = y.copy()
        for u in subj_ids:
            idx = idx_by_subj[u]
            y_perm[idx] = rng.permutation(y[idx])
        try:
            null[i] = roc_auc_score(y_perm, s)
        except ValueError:
            null[i] = 0.5
    null_dev = np.abs(null - 0.5)
    p = float((np.sum(null_dev >= obs_dev) + 1) / (n_iter + 1))
    return dict(auc_obs=auc_obs, p_value=p, null_mean=float(null.mean()),
                null_lo95=float(np.percentile(null, 2.5)),
                null_hi95=float(np.percentile(null, 97.5)),
                n=int(len(s)))


# ── small-n direction / equivalence ───────────────────────────────────────────

def sign_test_direction(per_subject_stat: Sequence[float], null: float = 0.5) -> dict:
    """Exact binomial test that a per-subject statistic sits on one side of `null`.

    Use for "k of 6 subjects have AUC > 0.5" or "> 0 effect". Reports the count
    and the exact two-sided p. At n = 6 you need 6/6 on one side to reach p<0.05
    (p = 0.031); 5/6 gives p = 0.22 — so "5 of 6 subjects showed the effect" is
    *not* significant, which the direction-dependent SWA result must reckon with.
    """
    from scipy.stats import binomtest
    x = np.asarray(per_subject_stat, float)
    x = x[np.isfinite(x)]
    n = len(x)
    n_above = int(np.sum(x > null))
    n_eff = int(np.sum(x != null))
    if n_eff == 0:
        return dict(n=n, n_above=n_above, n_effective=0, p_value=np.nan,
                    min_n_for_p05=np.nan)
    p = binomtest(n_above, n_eff, 0.5).pvalue
    # smallest unanimous count reaching p<0.05
    min_n = None
    for k in range(1, n_eff + 1):
        if binomtest(k, k, 0.5).pvalue < 0.05:
            min_n = k
            break
    return dict(n=n, n_above=n_above, n_effective=n_eff,
                p_value=float(p), min_n_for_p05=min_n)


def subject_mean_ci(
    per_subject_stat: Sequence[float],
    n_boot: int = 5000,
    seed: int = 0,
    ci: float = 95.0,
) -> dict:
    """Bootstrap CI for the cohort mean of a per-subject statistic.

    For a negative result, report this instead of "p > 0.05": e.g. the cohort
    mean EEG-CAP correlation is 0.015 with 95% CI [−0.02, +0.05], which *excludes*
    any correlation above 0.1 — a bounded, publishable statement of absence. `n`
    is the number of subjects, so the CI reflects the real unit of analysis.
    """
    x = np.asarray(per_subject_stat, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return dict(mean=float(x.mean()) if n else np.nan, ci_low=np.nan,
                    ci_high=np.nan, n=int(n))
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(x, size=n, replace=True).mean()
                     for _ in range(n_boot)])
    lo = (100 - ci) / 2
    return dict(mean=float(x.mean()),
                ci_low=float(np.percentile(boot, lo)),
                ci_high=float(np.percentile(boot, 100 - lo)),
                n=int(n))
