"""Driver: anchor the sleep-mask clock to wall time and merge with Muse.

    cd ~/Documents/01_Sleep-Analysis/optical_control_sync && python run_sync.py

Writes into this folder (never into the raw-data folder):
    sync_report.json / sync_report.md     anchor constants + QC numbers
    taps_sleepmask.csv                    the six tap pulses, both clocks
    mask_realtime_111Hz.csv.gz            full mask stream with a real-time column
    synced_mask_muse_10Hz.csv.gz          both devices on one 10 Hz wall-clock grid
    fig1_tap_anchor.png, fig2_synced_overview.png
"""

import json
import os

import numpy as np
import pandas as pd

import sync_mask_muse as S

OUT = os.path.dirname(os.path.abspath(__file__))


def main(make_figs=True, refine=True):
    # ---- mask: load (adds head tilt), check timing, find taps, set real time
    sm = S.load_sleepmask()
    timing = S.sleepmask_timing(sm)
    taps = S.detect_taps(sm)
    anchor_tap = S.anchor_from_taps(taps)
    sm = S.apply_clock(sm, anchor_tap)

    # ---- muse: load, check timing
    muse = S.load_muse()
    mtiming = S.muse_timing(muse)

    # ---- stage 2: refine the offset on large head-movement events
    upd = S.muse_accel_updates()
    ref = S.refine_sync_by_motion_events(sm, upd, n_events=200, min_sep_s=8.0,
                                         half_window_s=3.0, min_r=0.45,
                                         height_mult=5.0) if refine else dict(converged=False)
    anchor = S.refine_anchor(anchor_tap, ref)
    if anchor.get("refinement_lag_s"):
        sm = S.apply_clock(sm.drop(columns=["time"]), anchor)
    ref["events"].to_csv(os.path.join(OUT, "sync_motion_events.csv"), index=False)

    taps_out = taps.copy()
    taps_out["time"] = pd.to_datetime((taps_out["t_dev"].values + anchor["offset_s"]) * 1e9)
    taps_out.to_csv(os.path.join(OUT, "taps_sleepmask.csv"), index=False)

    # ---- independent check, on the clock actually being shipped
    cc = S.motion_cross_correlation(sm, muse, bin_s=1.0)

    # ---- outputs on real time
    cols = ["time", "timeMS"] + S.SM_CHANNELS + S.SM_ACC + S.SM_TILT
    sm[cols].to_csv(os.path.join(OUT, "mask_realtime_111Hz.csv.gz"),
                    index=False, compression="gzip",
                    date_format="%Y-%m-%d %H:%M:%S.%f")
    both = S.resample_common(sm, muse, fs=10.0)
    both.to_csv(os.path.join(OUT, "synced_mask_muse_10Hz.csv.gz"),
                index=False, compression="gzip",
                date_format="%Y-%m-%d %H:%M:%S.%f", float_format="%.5g")

    report = dict(
        mask_file=os.path.basename(S.SM_PATH), muse_file=os.path.basename(S.MUSE_PATH),
        mask_timing=timing, muse_timing=mtiming,
        tap_markers_reported=[m[0] for m in S.TAP_MARKERS],
        clock=dict(model="real_time = epoch(offset_s) + timeMS/1000",
                   offset_s=anchor["offset_s"],
                   offset_s_tap_only=anchor_tap["offset_s"],
                   offset_disagreement_between_bursts_s=anchor["offset_spread_s"],
                   mask_power_on=str(pd.to_datetime(anchor["offset_s"] * 1e9)),
                   mask_stream_start=str(sm["time"].iloc[0]),
                   mask_stream_end=str(sm["time"].iloc[-1])),
        refinement=({k: ref[k] for k in ("n_used", "lag_median_s", "lag_mad_s",
                                         "lag_robust_sd_s", "lag_se_s", "lag_iqr_s",
                                         "r_median")}
                    | dict(n_candidate_events=int(len(ref["events"])))
                    if ref.get("converged") else dict(converged=False)),
        per_burst=anchor["per_burst"].drop(columns=["offset_s"]).to_dict("records"),
        overlap=dict(start=cc["overlap_start"], end=cc["overlap_end"],
                     minutes=round((pd.Timestamp(cc["overlap_end"]) -
                                    pd.Timestamp(cc["overlap_start"])).total_seconds() / 60, 2)),
        motion_check={k: cc[k] for k in ("n_bins", "bin_s", "r_at_zero", "r_peak",
                                         "lag_peak_s", "r_background_p95")},
        synced_10Hz=dict(rows=int(len(both)), columns=list(both.columns)),
    )
    with open(os.path.join(OUT, "sync_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    import make_report
    make_report.write(report, taps_out, OUT)

    if make_figs:
        import figures
        figures.fig1_tap_anchor(sm, taps_out, anchor, OUT)
        figures.fig2_synced_overview(both, cc, taps_out, OUT)
        if ref.get("converged"):
            figures.fig3_motion_refinement(ref, sm, upd, OUT)

        import spectra
        spectra.fig4_mask_spectra(sm, OUT)
        spectra.fig5_muse_eeg_spectra(muse, OUT)
        spectra.fig6_optics_spectra(muse, OUT)
        spectra.fig7_aux_spectra(muse, OUT)
        spectra.band_table(sm, muse, OUT)
    return sm, muse, both, report, cc, taps_out, ref


if __name__ == "__main__":
    out = main()
    rep = out[3]
    print(json.dumps(rep["clock"], indent=2))
    print(json.dumps(rep["refinement"], indent=2, default=str))
    print(json.dumps(rep["motion_check"], indent=2))
