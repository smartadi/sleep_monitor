"""EEG-scored arousal counts and arousal index, per overnight recording.

Answers Jae's 2026-08-27 question: arousals per hour, or total, for each sleep.

Counts come from the PSG's own scoring -- the 'Classification Arousal' file that
sits beside the sleep profile -- so these are the technologist's cortical
arousals, not anything derived from the mask.

Two conventions worth stating, because they change the number:

  THE AROUSAL INDEX IS PER HOUR OF SLEEP, NOT PER HOUR OF RECORDING. That is the
  AASM definition and the one clinical norms are quoted against, so it is the
  figure to compare with published values. Recordings here hold 4.1-8.7 h of
  clock time but less sleep than that, and the two indices differ by however
  much wake the night contained. Both are reported.

  SUBTYPES ARE SEPARATED. The scorer labels arousals as spontaneous,
  respiratory-event-related, or limb-movement-related. A respiratory arousal
  index speaks to sleep-disordered breathing while a spontaneous one speaks to
  sleep fragmentation, and summing them hides which is driving a night.

Autonomic (pleth-derived) arousals are counted alongside, since the same PSG
scores them and they are the non-cortical comparison.

Run from the repo root:
    .venv/Scripts/python.exe analysis/swa_validation/arousal_index.py

Outputs -> reports/psg/arousal_counts.csv
           writeup/figures/prof_metrics/arousal_index.png
           writeup/figures/prof_metrics/arousal_table.png
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sleep_monitor.loader import (                                   # noqa: E402
    load_session, load_sleep_profile, load_arousals, load_autonomic_arousals)
from sleep_monitor.sessions import SESSION_META                       # noqa: E402

AGE = ROOT / "analysis" / "rates" / "outputs" / "k_vs_age_per_subject.csv"
CAP_EVENTS = ROOT / "reports" / "swa_validation" / "cap_events" / "cap_arousal_events.csv"
OUT = ROOT / "reports" / "psg"
FIG = ROOT / "writeup" / "figures" / "prof_metrics"
for p in (OUT, FIG):
    p.mkdir(parents=True, exist_ok=True)

EPOCH_HR = 30.0 / 3600.0
SLEEP_CODES = {0, 1, 2, 3}            # REM, N3, N2, N1 -- everything but wake
STAGE = {4: "Wake", 3: "N1", 2: "N2", 1: "N3", 0: "REM"}

C_INK = "#1B2A41"
C_MUTED = "#5A6472"
C_FAINT = "#D6DBE1"
# The scorer's own vocabulary, verbatim, and the grouping applied to it. Folding
# is done on exact labels rather than substrings: an earlier substring rule sent
# "SpO2 Arousal" and "Cardiac Arousal" -- 230 events, 8.5% of the cohort -- into
# the spontaneous bucket, which inflated spontaneous and understated respiratory.
# SpO2 arousals are desaturation-related and belong with respiratory; PLM is a
# limb subtype; cardiac gets its own column rather than a guess.
LABEL_GROUP = {
    "Arousal": "spontaneous",
    "Respiratory Arousal": "respiratory",
    "SpO2 Arousal": "respiratory",
    "LM Arousal": "limb",
    "PLM Arousal": "limb",
    "Cardiac Arousal": "cardiac",
}
BY_TYPE = {"spontaneous": "#2980B9", "respiratory": "#C0392B",
           "limb": "#E67E22", "cardiac": "#8E44AD"}

MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 11,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": C_INK, "text.color": C_INK,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED, "axes.edgecolor": C_MUTED,
    "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


_UNKNOWN = set()


def classify(label):
    """Group a scorer label. Unknown labels are reported, never silently folded."""
    t = (label or "").strip()
    if t in LABEL_GROUP:
        return LABEL_GROUP[t]
    _UNKNOWN.add(t)
    return "other"


def stage_at(t_hr, prof):
    """Scored stage code at each arousal onset."""
    pt = np.asarray(prof["t_ep_hr"], float)
    pc = np.asarray(prof["codes"], int)
    j = np.searchsorted(pt, np.asarray(t_hr, float)) - 1
    out = np.full(len(j), -1)
    ok = (j >= 0) & (j < len(pc))
    out[ok] = pc[j[ok]]
    return out


def build():
    rows = []
    for idx, meta in enumerate(SESSION_META):
        s = load_session(idx)
        prof = load_sleep_profile(s)
        ar = load_arousals(s)
        auto = load_autonomic_arousals(s)
        label = s.meta["label"]

        if prof is None:
            print("  %s -- no sleep profile, skipped" % label)
            continue

        codes = np.asarray(prof["codes"], int)
        rec_hr = float(s.time_hr[-1])
        tst_hr = float(np.isin(codes, list(SLEEP_CODES)).sum() * EPOCH_HR)

        rec = {"session": label, "subject": label[:2],
               "recording_hr": rec_hr, "tst_hr": tst_hr,
               "sleep_efficiency_pct": 100.0 * tst_hr / rec_hr if rec_hr else np.nan}

        if ar is None:
            print("  %s -- no arousal file" % label)
            rec.update(n_arousals=np.nan, arousal_index=np.nan)
        else:
            st = stage_at(ar["start_hr"], prof)
            in_sleep = np.isin(st, list(SLEEP_CODES))
            kinds = np.array([classify(t) for t in ar["types"]])

            rec["n_arousals"] = int(in_sleep.sum())
            rec["n_arousals_all"] = int(len(st))
            rec["arousal_index"] = rec["n_arousals"] / tst_hr if tst_hr else np.nan
            rec["per_recording_hour"] = rec["n_arousals"] / rec_hr if rec_hr else np.nan
            rec["median_duration_s"] = float(np.median(ar["duration_s"][in_sleep])) \
                if in_sleep.any() else np.nan
            for k in list(BY_TYPE) + ["other"]:
                n = int(((kinds == k) & in_sleep).sum())
                rec["n_" + k] = n
                rec["idx_" + k] = n / tst_hr if tst_hr else np.nan
            for lab in LABEL_GROUP:                       # raw counts, unfolded
                rec["raw_" + lab.replace(" ", "_")] = int(
                    ((np.array(ar["types"]) == lab) & in_sleep).sum())
            for code, name in STAGE.items():
                if code in SLEEP_CODES:
                    rec["n_in_" + name] = int((st == code).sum())

        if auto is not None:
            sta = stage_at(auto["start_hr"], prof)
            n_auto = int(np.isin(sta, list(SLEEP_CODES)).sum())
            rec["n_autonomic"] = n_auto
            rec["autonomic_index"] = n_auto / tst_hr if tst_hr else np.nan

        rows.append(rec)
        print("  %-6s  TST %4.1f h   arousals %4s   index %5.1f /h"
              % (label, tst_hr, rec.get("n_arousals", "--"),
                 rec.get("arousal_index", np.nan)))
    return pd.DataFrame(rows)


def add_cap(t):
    """CAP-detected arousal events on the same per-hour-of-sleep footing.

    Two rates, because they answer different questions. The KEPT rate counts
    events surviving the four artifact gates -- the conservative inventory. The
    DETECTED rate counts every transient before gating, which is the fairer
    comparison against the PSG: scored arousals are movement-coupled almost by
    definition, so the head-motion gate removes most of the events where the two
    instruments agree, and the kept rate under-counts by construction.

    Both are restricted to scored sleep, matching the PSG index.
    """
    if not CAP_EVENTS.exists():
        print("  no CAP event table -- comparison skipped")
        return t
    e = pd.read_csv(CAP_EVENTS)
    in_sleep = e.stage.isin(["N1", "N2", "N3", "REM"])
    g = (e[in_sleep].groupby("session")
         .agg(cap_detected=("kept", "size"), cap_kept=("kept", "sum"))
         .reset_index())
    t = t.merge(g, on="session", how="left")
    t["cap_kept_index"] = t.cap_kept / t.tst_hr
    t["cap_detected_index"] = t.cap_detected / t.tst_hr
    return t


def fig_cap_vs_psg(t):
    """The same measure -- arousals per hour of sleep -- from both instruments."""
    from scipy import stats as _st
    t = t.sort_values("session")
    x = np.arange(len(t))
    fig, ax = plt.subplots(1, 3, figsize=(250 * MM, 92 * MM))

    a = ax[0]
    a.bar(x - 0.21, t.arousal_index, width=0.42, color="#2980B9",
          label="PSG scored (left axis)")
    a.set_ylabel("PSG arousals per hour of sleep")
    a.set_xticks(x); a.set_xticklabels(t.session, rotation=90, fontsize=8)
    b = a.twinx()
    b.plot(x, t.cap_detected_index, "o-", color="#E67E22", lw=1.4, ms=5,
           label="CAP detected (right axis)")
    b.plot(x, t.cap_kept_index, "s--", color="#C0392B", lw=1.2, ms=4,
           label="CAP kept after gating (right axis)")
    b.set_ylabel("CAP events per hour of sleep")
    b.spines["top"].set_visible(False)
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    a.set_title("A  ·  per night, both instruments", loc="left", fontsize=10)

    c = ax[1]
    for col, colr, lab in [("cap_detected_index", "#E67E22", "detected"),
                           ("cap_kept_index", "#C0392B", "kept after gating")]:
        m = np.isfinite(t[col]) & np.isfinite(t.arousal_index)
        r = _st.spearmanr(t.arousal_index[m], t[col][m]).statistic
        c.scatter(t.arousal_index[m], t[col][m], s=38, color=colr, alpha=0.85,
                  label="CAP %s   ρ=%+.2f" % (lab, r))
    c.set_xlabel("PSG arousal index  (per hour of sleep)")
    c.set_ylabel("CAP event rate  (per hour of sleep)")
    c.set_title("B  ·  do they rank the nights the same way?", loc="left", fontsize=10)
    c.legend(fontsize=8, loc="upper left")

    # positive control: the PSG's own non-cortical arousal channel. If the null in
    # panel B were just n = 12 nights or a noisy index, this would be null too.
    d = ax[2]
    m = np.isfinite(t.autonomic_index) & np.isfinite(t.arousal_index)
    r = _st.spearmanr(t.arousal_index[m], t.autonomic_index[m]).statistic
    d.scatter(t.arousal_index[m], t.autonomic_index[m], s=38, color="#16A085",
              alpha=0.85, label="pleth autonomic   ρ=%+.2f" % r)
    lim = [0, max(t.arousal_index.max(), t.autonomic_index.max()) * 1.08]
    d.plot(lim, lim, ls="--", lw=0.8, color=C_MUTED)
    d.set_xlim(lim); d.set_ylim(lim)
    d.set_xlabel("PSG arousal index  (per hour of sleep)")
    d.set_ylabel("autonomic index  (per hour of sleep)")
    d.set_title("C  ·  positive control, same nights", loc="left", fontsize=10)
    d.legend(fontsize=8, loc="upper left")

    for a_ in ax:
        a_.grid(axis="y", color=C_FAINT, lw=0.5)
        a_.set_axisbelow(True)
    fig.suptitle("Cortical arousal — PSG scoring against the capacitive mask",
                 fontsize=11.5, x=0.055, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = FIG / "arousal_cap_vs_psg.png"
    fig.savefig(p); plt.close(fig); print("  %s" % p.name)


def fig_index(t):
    t = t.sort_values("session")
    x = np.arange(len(t))
    fig, ax = plt.subplots(1, 2, figsize=(250 * MM, 100 * MM))

    bottom = np.zeros(len(t))
    for k, c in BY_TYPE.items():
        v = t["idx_" + k.split()[0]].to_numpy(float)
        ax[0].bar(x, v, bottom=bottom, color=c, width=0.68, label=k)
        bottom += np.nan_to_num(v)
    ax[0].set_xticks(x); ax[0].set_xticklabels(t.session, rotation=90, fontsize=8)
    ax[0].set_ylabel("arousals per hour of sleep")
    ax[0].set_title("Arousal index by subtype", loc="left", fontsize=10)
    ax[0].legend(fontsize=8)

    ax[1].bar(x - 0.19, t.arousal_index, width=0.36, color="#2980B9",
              label="cortical (EEG-scored)")
    if "autonomic_index" in t:
        ax[1].bar(x + 0.19, t.autonomic_index, width=0.36, color="#16A085",
                  label="autonomic (pleth)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(t.session, rotation=90, fontsize=8)
    ax[1].set_ylabel("events per hour of sleep")
    ax[1].set_title("Cortical against autonomic arousals", loc="left", fontsize=10)
    ax[1].legend(fontsize=8)

    for a in ax:
        a.grid(axis="y", color=C_FAINT, lw=0.5)
        a.set_axisbelow(True)
    fig.suptitle("EEG-scored arousals, per night", fontsize=11.5, x=0.055, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = FIG / "arousal_index.png"
    fig.savefig(p); plt.close(fig); print("  %s" % p.name)


def fig_table(t):
    cols = ["session", "age", "psqi", "recording_hr", "tst_hr", "n_arousals",
            "arousal_index", "idx_spontaneous", "idx_respiratory", "idx_limb",
            "autonomic_index"]
    head = ["night", "age", "PSQI", "rec\n(h)", "TST\n(h)", "arousals\n(n)",
            "index\n(/h sleep)", "spont.\n(/h)", "resp.\n(/h)", "limb\n(/h)",
            "autonomic\n(/h)"]
    fmt = ["%s", "%.0f", "%.0f", "%.1f", "%.1f", "%.0f", "%.1f", "%.1f", "%.1f",
           "%.1f", "%.1f"]
    d = t.sort_values("session")
    cells = [[f % v if not (isinstance(v, float) and np.isnan(v)) else "—"
              for f, v in zip(fmt, row)] for row in d[cols].to_numpy()]

    fig, ax = plt.subplots(figsize=(250 * MM, 118 * MM))
    ax.axis("off")
    tb = ax.table(cellText=cells, colLabels=head, loc="center", cellLoc="right")
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.75)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(C_FAINT); cell.set_linewidth(0.6)
        if r == 0:
            cell.set_text_props(color=C_INK, fontweight="bold")
            cell.set_facecolor("#EFF1F5")
        elif r % 2 == 0:
            cell.set_facecolor("#FAFBFC")
        if c == 0:
            cell.set_text_props(ha="left")
    ax.set_title("EEG-scored arousals — all twelve nights", loc="left",
                 fontsize=11.5, pad=16)
    p = FIG / "arousal_table.png"
    fig.savefig(p); plt.close(fig); print("  %s" % p.name)


def main():
    print("Reading PSG arousal scoring")
    t = build()
    t = add_cap(t)
    a = pd.read_csv(AGE)[["subj", "age", "psqi"]].drop_duplicates("subj")
    t = t.merge(a, left_on="subject", right_on="subj", how="left").drop(columns="subj")
    t.to_csv(OUT / "arousal_counts.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nPer night\n")
    print(t[["session", "age", "psqi", "recording_hr", "tst_hr", "n_arousals",
             "arousal_index", "idx_spontaneous", "idx_respiratory", "idx_limb",
             "idx_cardiac", "autonomic_index", "cap_detected_index",
             "cap_kept_index"]].round(2).to_string(index=False))

    print("\nFigures")
    fig_index(t)
    fig_cap_vs_psg(t)
    fig_table(t)
    print("\ntable -> %s" % (OUT / "arousal_counts.csv"))


if __name__ == "__main__":
    main()
