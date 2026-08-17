"""Can any function of the capacitive epoch predict the reference rate?

Every within-night result in the paper is mediated by a rate *estimator*: a
scalar summary of the epoch. If the information about this epoch's rate lives in
the shape of its spectrum rather than in a peak count, every estimator fails
while the information is present and untouched. Seven estimators, four channels,
fusion, CWT ridges and Viterbi tracking are all the same kind of summary.

Two facts sharpen the question. Averaging to ten minutes does not improve the
correlation, so the failure is not noise a longer window would remove. And the
cardiac band carries ECG-specific amplitude information (see cap_psg_coupling),
so there is something there to decode.

This is the decisive test. Build a rich per-epoch feature vector from the
capacitive channels only, and ask whether a regressor can predict the reference
rate *within a night*, blocked-CV so no epoch is predicted by its neighbours,
scored against the same circular-shift null the tracking analysis uses.

  it fails     the negative becomes a property of the signal at this window
               length, not of the estimator family -- a much stronger 5.2
  it succeeds  the negative was about peak counting, and the paper gains a
               positive result

Blocked cross-validation matters: rate series are autocorrelated over minutes, so
random k-fold would let the model memorise a neighbour and report a correlation
that is really interpolation. Contiguous blocks are held out whole.

Outputs -> reports/rates/decoder/decoder_results.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sleep_monitor.config import FS, RESP_LO, RESP_HI, CARD_LO, CARD_HI  # noqa: E402
from sleep_monitor.loader import load_session                             # noqa: E402
from sleep_monitor.sessions import SESSION_META                           # noqa: E402

OUT = ROOT / "reports" / "rates" / "decoder"
OUT.mkdir(parents=True, exist_ok=True)

EPOCH_S = 30.0
CHANNELS = ["CH", "CLE", "CRE", "CLE-CRE"]
BANDS = {"resp": (RESP_LO, RESP_HI), "card": (CARD_LO, CARD_HI)}
N_BLOCKS = 5


def epoch_features(x, lo, hi, fs=FS):
    """Spectral shape of one epoch, not a single rate summary."""
    x = x - x.mean()
    if np.std(x) < 1e-12:
        return None
    f, p = signal.welch(x, fs=fs, nperseg=min(len(x), int(fs * 10)))
    m = (f >= lo) & (f <= hi)
    if m.sum() < 3:
        return None
    pb, fb = p[m], f[m]
    tot = pb.sum() + 1e-20
    pn = pb / tot
    # spectral shape: where the mass is, how concentrated, how skewed
    centroid = float((fb * pn).sum())
    peak_f = float(fb[np.argmax(pb)])
    spread = float(np.sqrt(((fb - centroid) ** 2 * pn).sum()))
    entropy = float(-(pn * np.log(pn + 1e-20)).sum())
    flat = float(stats.gmean(pb + 1e-20) / (pb.mean() + 1e-20))
    top = float(pb.max() / (pb.mean() + 1e-20))
    # quartiles of the in-band spectrum
    c = np.cumsum(pn)
    q = [float(fb[np.searchsorted(c, t)]) for t in (0.25, 0.5, 0.75)]
    # autocorrelation of the band-limited signal
    b, a = signal.butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    y = signal.filtfilt(b, a, x)
    ac = np.correlate(y, y, "full")[len(y) - 1:]
    ac = ac / (ac[0] + 1e-20)
    lo_lag, hi_lag = int(fs / hi), int(fs / lo)
    seg = ac[lo_lag:hi_lag] if hi_lag > lo_lag + 2 else np.array([0.0])
    ac_peak = float(seg.max())
    ac_lag = float((lo_lag + np.argmax(seg)) / fs)
    env = np.abs(signal.hilbert(y))
    return [centroid, peak_f, spread, entropy, flat, top, *q,
            ac_peak, ac_lag, float(np.log(tot)),
            float(env.mean()), float(env.std() / (env.mean() + 1e-20))]


def build(band):
    lo, hi = BANDS[band]
    rows = []
    for idx, meta in enumerate(SESSION_META):
        s = load_session(idx)
        cap = dict(s.cap)
        cap["CLE-CRE"] = cap["CLE"] - cap["CRE"]
        gt = pd.read_parquet(ROOT / "artifacts" / "rate_rerun_phase_a.parquet")
        gt = gt[(gt.session == s.meta["label"]) & (gt.band == band)
                & (gt.channel == "CRE")][["epoch", "gt_hz"]].dropna()
        gmap = dict(zip(gt.epoch.astype(int), gt.gt_hz.values * 60.0))
        win = int(EPOCH_S * FS)
        for e in range(len(s.cap["CH"]) // win):
            if e not in gmap:
                continue
            feats = []
            ok = True
            for ch in CHANNELS:
                fv = epoch_features(np.asarray(cap[ch][e * win:(e + 1) * win], float), lo, hi)
                if fv is None:
                    ok = False
                    break
                feats.extend(fv)
            if ok:
                rows.append([s.meta["label"], e, gmap[e]] + feats)
        print("  %s %s: %d epochs" % (s.meta["label"], band, len(rows)))
    cols = ["session", "epoch", "target"] + [
        "%s_%d" % (ch, i) for ch in CHANNELS for i in range(14)]
    return pd.DataFrame(rows, columns=cols)


def blocked_cv(X, y, model):
    """Predict each contiguous block from the others."""
    n = len(y)
    edges = np.linspace(0, n, N_BLOCKS + 1).astype(int)
    pred = np.full(n, np.nan)
    for i in range(N_BLOCKS):
        te = np.zeros(n, bool)
        te[edges[i]:edges[i + 1]] = True
        if te.sum() < 5 or (~te).sum() < 20:
            continue
        m = model()
        m.fit(X[~te], y[~te])
        pred[te] = m.predict(X[te])
    return pred


def main():
    results = []
    for band in BANDS:
        d = build(band)
        d.to_parquet(OUT / ("features_%s.parquet" % band), index=False)
        feat_cols = [c for c in d.columns if c not in ("session", "epoch", "target")]
        for name, model in (("ridge", lambda: RidgeCV(alphas=np.logspace(-3, 3, 13))),
                            ("gbm", lambda: HistGradientBoostingRegressor(
                                max_depth=3, max_iter=200, learning_rate=0.05))):
            rng = np.random.default_rng(0)
            for sess, g in d.groupby("session"):
                g = g.sort_values("epoch")
                X = g[feat_cols].to_numpy(float)
                y = g["target"].to_numpy(float)
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                if len(y) < 100 or np.std(y) < 1e-9:
                    continue
                pred = blocked_cv(X, y, model)
                ok = np.isfinite(pred)
                if ok.sum() < 50:
                    continue
                r = float(np.corrcoef(pred[ok], y[ok])[0, 1])
                mae = float(np.mean(np.abs(pred[ok] - y[ok])))
                # circular-shift null on the prediction
                nulls = [float(np.corrcoef(np.roll(pred[ok], int(rng.integers(1, ok.sum()))),
                                           y[ok])[0, 1]) for _ in range(200)]
                results.append({"band": band, "model": name, "session": sess,
                                "r": r, "mae": mae, "ref_sd": float(np.std(y[ok])),
                                "null_hi": float(np.percentile(nulls, 97.5)),
                                "beats_null": bool(r > np.percentile(nulls, 97.5))})
    res = pd.DataFrame(results)
    res.to_csv(OUT / "decoder_results.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nBlocked-CV within-night decoding, capacitive features only\n")
    for (band, model), g in res.groupby(["band", "model"]):
        p = stats.wilcoxon(g.r).pvalue if len(g) > 5 else np.nan
        print("  %-5s %-6s  median r %+.3f   %d/%d nights beat the shift null   "
              "median MAE %.2f (reference SD %.2f)   Wilcoxon p = %.3f"
              % (band, model, g.r.median(), g.beats_null.sum(), len(g),
                 g.mae.median(), g.ref_sd.median(), p))
    print("\nwrote %s" % (OUT / "decoder_results.csv"))


if __name__ == "__main__":
    main()
