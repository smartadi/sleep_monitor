# Code Changelog

Records all code changes to library modules, scripts, and notebooks.

---

## 2026-07-23

- **Added** `analysis/delta_onset/` (new scoped workspace, with `CLAUDE.md`) — investigation of the professor's hypothesis that at EEG delta-burst onset a *preceding* event appears in CAP bands 0–0.5 / 0.5–1 / 1–3 Hz (CLE/CRE/CH). `delta_onset_detection.py` builds and validates the delta-onset trigger: Schmitt-trigger burst detector on the EEG delta (0.5–4 Hz) Hilbert envelope (per-session robust NREM baseline med±MAD, high=+2·MAD/low=+0.5·MAD, ≥4 s sustained), onset walked back to the burst rising edge, ≥25 s refractory, kept only if in NREM with a motion-clean −30→0 s pre-window. Caches EEG delta envelope + per-sample stage codes + motion mask to `$TEMP/delta_onset_cache` for fast re-tuning. Writes per-session onset npz, `delta_onsets_summary.csv`, and overview + single-event-gallery figures to `analysis/delta_onset/outputs/`. CLI `-s/--session`, `--show`. Run: `.venv/Scripts/python.exe analysis/delta_onset/delta_onset_detection.py`. Cohort: 11–197 onsets/session, 100% NREM, N2-dominant.

- **Changed** `analysis/mean_value/channel_evolution.py` — (1) baseline-velocity row (D) now reports a **non-artifact velocity**: both accelerometer-motion epochs (top-decile) AND large baseline **step** artifacts (electrode coupling loss/regain, detected from the mean via a robust MAD threshold since the accelerometer does NOT flag them) are blanked (NaN) before autoscaling, so the row shows the slow physiological drift instead of ±20,000 a.u./hr motion/coupling spikes (esp. poor-contact S6). Step mask dilated ±18 blocks (~3 min) to cover the smoothing bump. (2) Spectrogram row (F) now annotates the **respiratory (0.1–0.5 Hz)** and **cardiac (0.5–3.0 Hz)** bands with dashed lines + labels.

## 2026-07-21

- **Added** `analysis/mean_value/channel_evolution.py` — per-session, per-channel journal figure of overnight signal evolution. One stacked panel per channel (CLE, CRE, CH) with rows A–F: hypnogram strip, mean value, variance, smoothed baseline velocity, motion (accelerometer std), and a 0–5 Hz spectrogram with dB colorbar. The raw channel is zero-phase low-pass filtered at 10 Hz (Butterworth-4) before mean/variance so those reflect physiological-band content, not the >10 Hz electronic floor. Mean/variance/motion from 10 s blocks; velocity = smoothed d(mean)/dt (3-min baseline smooth, 2-min velocity smooth); every data row autoscaled to robust 1–99th-pct limits so evolution isn't squished; faint per-stage shading behind line rows. CLI `--session <label>` or `--all`. Outputs `writeup/figures/channel_evolution/<SESSION>_<channel>.png` (12 sessions × 3 channels = 36). Run: `.venv/Scripts/python.exe analysis/mean_value/channel_evolution.py --all`.
- **Added** `analysis/mean_value/abs_mean_vs_stage.py` — absolute (native a.u.) CAP mean-value vs sleep-stage analysis, the complement to the z-scored slow-mean scripts. Consumes the raw a.u. columns already in `reports/mean_value/mean_value_epochs.csv` (no raw reload for stats); controls the between-session DC offset by subtracting each channel's per-session Wake median (preserves a.u. scale, unlike z-score). Computes: between-session offset spread, within-night drift (last-30min − first-30min), per-session-centered per-stage medians + Kruskal–Wallis, and per-subject Wake/N3 sign consistency. Loads the 3 representative sessions (S1N1/S3N1/S5N1) for a raw-a.u. trace + low-freq spectrogram figure. Outputs `reports/mean_value/abs_mean_{scale, stage_au, subject_direction}.csv` and figs `notebooks/plots/mean_value/abs_{baseline_by_session, stage_boxplot_au, scale_comparison, trace_<SESSION>}.png`. Run: `.venv/Scripts/python.exe analysis/mean_value/abs_mean_vs_stage.py`.

---

## 2026-07-16

- **Added** `writeup/figures/signal_validation/psd_db_per_session.py` — simple frequency-vs-dB power-spectrum plot of the CAP signal, per session. x = frequency (Hz, 0–20), y = power in dB (`10*log10` of the full-night Welch PSD); CLE−CRE differential bold + CH/CLE/CRE faint; resp (0.1–0.5) and cardiac (0.5–3.0 Hz) bands shaded. Reuses the cached PSDs in `signal_characterization_cache.pkl` (no raw reload). Outputs a 4×3 all-sessions grid `psd_db_all_sessions.png` + one PNG per session under `psd_db/`. Run: `.venv/Scripts/python.exe writeup/figures/signal_validation/psd_db_per_session.py`.
- **Changed** `sleep_monitor/harmonics.py` — added a backward-compatible `min_freq` parameter to `detect_persistent_ridges` (default 0.0, so existing callers are unaffected). It band-limits the Welch frequency mask (`(f<=max_freq)&(f>=min_freq)`) and the ridge-frequency floor (`min_ridge_freq_floor = max(min_freq, 2*df)`), enabling per-band ridge detection (respiratory 0.1–0.5 Hz, cardiac 0.5–3.0 Hz).
- **Added** `analysis/slow_wave/band_ridge_analysis.py` — runs `detect_persistent_ridges` twice per session×channel, band-restricted to the respiratory (0.1–0.5 Hz, 30 s periodogram, jump 0.05) and cardiac (0.5–3.0 Hz, 8 s Welch, jump 0.10) bands, over all 12 sessions × 3 CAP channels; per-epoch ridge features (n_ridges, min/mean ridge freq, total power, freq spread, harmonic groups, ridge_present) aligned to PSG stage; KW + N3-vs-rest MWU + per-subject direction (pooled on CRE). Outputs `reports/slow_wave/{band_ridge_epochs.parquet (55,914 rows), band_ridge_stage_summary.csv}`, figs `writeup/figures/harmonics/{band_ridge_by_stage.png, band_ridge_overlay_S1N1.png}`. Run: `.venv/Scripts/python.exe analysis/slow_wave/band_ridge_analysis.py`.
- **Added** `analysis/slow_wave/band_ridge_figure.py` — reloads `band_ridge_epochs.parquet` (no recompute) and renders the paper figure `band_ridge_by_stage.png` (2 bands × {mean active ridges bar, total ridge power box, lowest ridge freq box} by stage, KW p-values) and prints the citable numbers used in §4.3. Run: `.venv/Scripts/python.exe analysis/slow_wave/band_ridge_figure.py`.
- **Edited** the canonical manuscript `writeup/main/CAP_sleep_mask_manuscript_main.docx` directly (unzip → edit `word/document.xml` via lxml → rezip via `zipfile`; no LibreOffice/pandoc on this machine). Per user request: removed the negative within-session rate-tracking section (Results §4.3, Methods §3.6, Discussion subsection, Figs 6–9, Table 4), the LOSO N3 classification subsection (Fig 12, Table 5, Methods sentence, Discussion subsection), and the SEC-vs-contact-EEG SWA negative section (Results §4.5, Methods §3.8, Discussion subsection, Figs 13–14, Table 6, uncited Lucey ref #30); trimmed the broadband harmonic/ridge Methods+Results and rewrote §4.3 as "Harmonic structure and band-restricted ridge features" with the resp/cardiac band results, swapping the embedded pooled-harmonic image (media/image12.png, rId17; cy set to 2926080 EMU for the 2700×1440 PNG) for `band_ridge_by_stage.png`. Fixed all cascading cross-refs and renumbered Methods 3.6–3.7 / Results 4.1–4.4 / Figures 1–8 / Discussion 5.1–5.5. Clean edit (no tracked changes — prior redline already accepted); XSD-validated (`validate.py`, PASSED, 629→475 paragraphs). Pre-edit backup `CAP_sleep_mask_manuscript_main_PRE_NEGATIVE_REMOVAL_backup_20260716.docx`.
- **Added** `analysis/spindles/spindle_lowband_subbands.py` — breaks the CAP 0–3 Hz spindle-onset bump into sub-bands 0–0.5 / 0.5–1.5 / 1.5–3 Hz (CH), onset-triggered average per N2 spindle, all 12 sessions. Same ±8 s window + own-baseline dB contrast as `spindle_lowband_detection.py` but finer STFT (nperseg=256 → 0.39 Hz bins) so the sub-bands resolve. Recomputes from raw sessions, caches onset-triggered curves to `analysis/spindles/outputs/spindle_lowband_subbands.npz` (pass `--recompute` to rebuild). Figure: grand-mean panel + 12-session grid, per-session title flags which sub-band peaks. **Finding: the bump is broadband across 0–3 Hz — all three sub-bands rise at onset** (grand-mean core dB: slow +0.47/11·12, mid +0.52/11·12, high +0.37/12·12); slow (0–0.5 Hz) has the largest, most sustained excursion, high (1.5–3 Hz) the flattest/earliest — consistent with a mechanical/hemodynamic transient, not a narrowband oscillation. Output `writeup/figures/spindles/fig_spindle_lowband_subbands.png`. Run: `.venv/Scripts/python.exe -m analysis.spindles.spindle_lowband_subbands`.
- **Added** `analysis/spindles/plot_spindle_lowband_persession_grid.py` — the most basic spindle finding, one panel per session (4×3 grid, all 12): pure onset-triggered average of CAP 0–3 Hz power (baseline-corrected) around N2 spindle centers, CH bold + CLE/CRE/CLE−CRE faint, shared axes. Every session shows the power rising to a peak at the spindle onset (t=0). Reads cached `spindle_lowband_detection.npz`/`.csv` (no recompute). Output `writeup/figures/spindles/fig_spindle_lowband_persession_grid.png`. Run: `.venv/Scripts/python.exe -m analysis.spindles.plot_spindle_lowband_persession_grid`.
- **Edited** the canonical manuscript `writeup/main/CAP_sleep_mask_manuscript_main.docx` directly (unzip → edit word/document.xml → rezip) to reflect the simplified Figure 2: swapped the embedded image (media/image3.png, rId8) from the old 3-panel fig7 to `fig2_inband_snr.png` (fixed the display aspect ratio cy 2348029→2759569 EMU for the 2179×1096 PNG), and rewrote all five SNR passages — Methods definition, Results paragraph (dropped "+11 to +27 / +4 to +13 dB vs 3.5–5 Hz floor", added broadband "signal <10 Hz vs noise 10–50 Hz, mean +12.6 dB, 11/12 positive"), Figure 2 caption, Discussion "Physiological band energy", and the appendix figure list. Clean edit (no tracked changes — prior redline already accepted); pre-edit backup saved as `CAP_sleep_mask_manuscript_main_PRE_SNR_backup_20260716.docx`. Verified: document.xml well-formed UTF-8, all new strings present / old absent, zip intact.
- **Added** `writeup/figures/signal_validation/inband_snr.py` — simplified, baseline-free replacement for the manuscript Figure 2 "in-band SNR analysis". Collapses the old 3-panel `fig7_inband_snr.png` (only 3 sessions) and the 3-panel `fig8_simplified_snr.png` into a single-panel bar chart over all 12 sessions. New noise definition (no baseline model needed): signal = full-night power below 10 Hz (where all CAP physiology lives), noise = power from 10 Hz to Nyquist (50 Hz); SNR = 10·log10(P_sig/P_noise) on the raw CLE−CRE full-night Welch PSD (Welch per-segment detrend suppresses the drifting DC baseline). Reads the existing `signal_characterization_cache.pkl` (PSDs already reach Nyquist), so no raw reload. Output `fig2_inband_snr.png` + `inband_snr_summary.csv`. Per-session SNR: mean +12.6 dB, median +11.8, range −0.3 (S5N2) to +22.4 (S4N1); positive in 11/12. **Changed** `scripts/generate_manuscript_docx.py` — Figure 2 now points at `fig2_inband_snr.png` with a rewritten caption + results paragraph; dropped the stale "+11 to +27 dB / +4 to +13 dB vs 3.5–5 Hz floor" resp/cardiac SNR numbers.
- **Added** `analysis/swa_validation/fultz_eeg_cap_impulse.py` — Fultz et al. (Science 2019) style lead-lag / impulse-response test of slow-delta EEG → slow CAP, per the professor's suggestion. Fultz found slow-delta EEG (0.2–4 Hz) envelope LEADS CSF inflow by ~6.4 s via a best-fit impulse response (not a zero-lag correlation); our prior SWA-validation only tested zero-lag spectral agreement (r≈0.015) so it could not have seen a delayed downstream coupling. Method (per session, NREM only): EEG bandpass 0.2–4 Hz → Hilbert envelope → smooth → decimate to 5 Hz (continuous); each CAP channel (CLE, CRE, CH, CLE−CRE OLS) decimated to 5 Hz then slow-bandpassed 0.01–0.1 Hz (Butterworth SOS, filtered continuously then NREM-sliced to avoid per-segment edge transients); contiguous NREM runs ≥120 s from the aligned sleep profile. FFT-based cross-correlation over ±40 s (positive lag = EEG leads CAP), circular-shift null (500×, 95% band), ridge (Tikhonov) impulse response EEG→CAP, and peak-locked average of the EEG envelope at slow-CAP peaks. Caches the 4 needed channels + profile per session to scratchpad `.npz` for fast reruns. First pass S2N2 (390 min NREM): all channels significant vs null but weak — CRE cleanest (|r|=0.15 at +2 s, IR r=0.22 ≈ Fultz's CV 0.22), CLE peak at +6.6 s (≈ Fultz 6.4 s) but |r|=0.08; lag sign inconsistent across channels, CAP-slow has no discrete ~0.05 Hz peak → not yet a clean replication (needs motion/arousal confound control + cohort). Output `analysis/swa_validation/outputs/fultz_eeg_cap_S2N2.png`. Run: `PYTHONPATH=. .venv/Scripts/python.exe analysis/swa_validation/fultz_eeg_cap_impulse.py --session 3`.
- **Added** `analysis/spindles/plot_spindle_lowband_allsessions.py` — all-12-session summary of the 0–3 Hz CAP spindle-onset bump, from the cached `spindle_lowband_detection.{npz,csv}` (no recompute). (A) all 12 CH onset-triggered curves + grand mean±SD; (B) per-session onset bump (core dB) bars + per-spindle detection-rate line vs matched-control chance; (C) grand-mean onset response per CAP channel + sigma-at-chance annotation. Result: the onset bump is positive in 12/12 sessions on CH (median +0.48 dB, range 0.15–1.25), all 4 CAP channels bump (+0.43–0.54 dB, CH strongest), while per-spindle detection stays marginal (48–55%). Output `writeup/figures/spindles/fig_spindle_lowband_allsessions.png`. Run: `.venv/Scripts/python.exe -m analysis.spindles.plot_spindle_lowband_allsessions`.
- **Added** `analysis/mean_value/flow_imbalance_vs_quality.py` — treats the <0.05 Hz L−R baseline (`vlf_CLE-CRE`) as a signed "flow direction" and tests the user hypothesis that one-sided flow across the night predicts worse sleep. Builds on the cached 30-s epoch table `reports/mean_value/mean_value_epochs.csv` (no raw reload). Per session, over the sleep period with top-decile-motion epochs dropped, computes offset-SENSITIVE metrics (`flow_bias`=median operating point, `flow_onesided`=|mean|/rms, `flow_offset_dom`=|median|/std) and offset-INVARIANT dynamics (`flow_skew`/`flow_absskew`, `flow_reversal`=median-crossings/hr, `flow_maxrun`=longest one-sided run); sleep-quality metrics (sleep_eff, WASO, SOL, awakenings, %N3, %REM, fragmentation) derived from hypnogram codes in the same table. Spearman grid (n=12), within-subject de-meaned Spearman (removes fixed per-mount offset → isolates night-to-night state), within-subject paired night check, and a flow-by-stage boxplot. Finds: absolute one-sidedness is offset-dominated (≈0.99 for all sessions → instrumental); offset-invariant dynamics only weakly trend the hypothesized way (p>0.19); signed `flow_bias` correlates strongly with quality (sleep_eff ρ=−0.76) and survives within-subject de-meaning but is exactly the mask-placement-confounded quantity. Outputs `reports/mean_value/{flow_imbalance_session.csv, flow_quality_corr.csv, flow_quality_within_subject.csv}` and figures under `notebooks/plots/mean_value/flow_*.png`, `scatter_*.png`. Run: `.venv/Scripts/python.exe analysis/mean_value/flow_imbalance_vs_quality.py`.
- **Added** `analysis/spindles/plot_spindle_trials_lowband.py` — per-trial (per-spindle-onset) view of the low-band (0–3 Hz) CAP response, showing the detection ON DATA. For one session, recomputes each N2 spindle's ±8 s CH low-band power trace (baseline-corrected, same short-time-spectrogram contrast as `spindle_lowband_detection.py`) plus the contact-EEG sigma trace marking the spindle. Figure: (A) onset-triggered average CH low-band power ±SEM over all N2 spindles (clean +1.2 dB bump at onset on S2N1); (B..) grid of individual spindle trials sorted strongest-first, EEG sigma (grey) + CH low-band (blue), ±1 s core shaded, per-trial dB + detected/not verdict — motion-artifact outliers (|dB|>12) excluded from the picks so examples show the physiological ~0.5–3 dB response. Honestly conveys the ~53% per-trial detection rate. Output `writeup/figures/spindles/fig_spindle_trials_lowband_<session>.png`. Run: `.venv/Scripts/python.exe -m analysis.spindles.plot_spindle_trials_lowband --session S2N1`.
- **Added** `analysis/spindles/spindle_lowband_detection.py` — per-spindle, onset-level DETECTION-RATE analysis of the low-band (0–3 Hz) CAP response to N2 sleep spindles. Extends the pooled `event_vs_control_auc` in `spindle_analysis.py` to score EACH spindle individually. Per spindle: ±8 s window, short-time spectrogram (nperseg=128, noverlap=96 — identical to `spindle_ersp.py`), band power averaged in dB across bins, then per-spindle dB = (mean core |t|<1 s) − (mean own-baseline |t|>5 s). DETECTION RULE: spindle "detected" if that dB > 0. Bands: low 0–3 Hz (primary, the validated bump), 0.5–3 Hz (env_c variant), sigma 11–16 Hz (built-in negative control). All 4 CAP channels (CLE, CRE, CLE−CRE, CH) + EEG reference, all 12 sessions, N2 spindles. Matched N2 controls (≥3 s from any spindle) scored identically → empirical chance reference. Loads raw sessions (per-spindle core/control windows are not cached). Outputs: 3-panel figure `writeup/figures/spindles/fig_spindle_lowband_detection.png` (A per-session low-band detection-rate bars per channel + chance line; B per-spindle effect-size distributions CH low-band vs sigma; C onset-triggered 0–3 Hz power average, CH bold); CSV `analysis/spindles/outputs/spindle_lowband_detection.csv` (per session + pooled: n_spindles, low-band + 0.5–3 Hz + sigma detection rate & mean dB per channel, empirical null); `spindle_lowband_detection.npz` (per-spindle dB arrays + triggered curves). Run: `.venv/Scripts/python.exe -m analysis.spindles.spindle_lowband_detection`.
- **Added** `analysis/slow_wave/resp_vs_swa_band_separation.py` — paper ridge/spectral section: separates the CAP (SEC) spectrum into the respiratory band (0.1–0.5 Hz) and the SWA/delta band (0.5–4 Hz, sub-bands SO 0.5–1 / delta_low 1–2 / delta_high 2–4), and shows the CAP delta/SWA band is cardiac/respiratory (harmonics + baseline wander), NOT cortical slow-wave. Reuses the cached full-night Welch PSDs from `writeup/figures/signal_validation/signal_characterization_cache.pkl` (no raw reload for the PSD decomposition); computes per-window band-power time courses fresh from raw CLE−CRE via `sleep_monitor.spectral.compute_band_power_ratios` (60 s window / 30 s step / 20 s Welch seg, accel motion-gated) for the across-night co-variation test. Metrics per session: resp/SWA and SO/delta_low/delta_high power fractions of 0.1–4 Hz; cardiac(0.5–3)/SWA overlap; spectral-trough separation ratio + trough freq; cardiac fundamental freq; fraction of SWA power on the cardiac harmonic comb (±0.08 Hz); and per-window Spearman ρ(delta,cardiac), ρ(delta,resp), ρ(delta_high,cardiac|resp). Cites the EEG-vs-CAP negative from `analysis/swa_validation/outputs/swa_validation_per_subject.csv`. One 4-panel figure (A mean PSD annotated resp vs SWA sub-bands + cardiac fundamental; B per-session power-fraction stacked bars; C log-log delta-vs-cardiac power scatter; D per-session ρ bars + EEG-negative box) → `writeup/figures/band_separation/resp_vs_swa_band_separation.png`; CSV `resp_vs_swa_band_separation.csv`. Run: `.venv/Scripts/python.exe analysis/slow_wave/resp_vs_swa_band_separation.py`.
- **Added** `writeup/figures/signal_validation/signal_characterization.py` — paper signal-validation block covering the three requested items over all 12 sessions, using only the `sleep_monitor` package + raw sessions. (1) **Simplified SNR:** a single per-60 s-window ratio `SNR_band(dB) = 10·log10(mean_PSD_in_band / mean_PSD_in_noise)` on raw broadband CLE−CRE, where each term is a per-Hz density (band power / band width) and the noise reference is a fixed physiology-free 5–10 Hz band, so the ratio is bandwidth-independent and easy to state in Methods. (2) **Mean-capacitance drift:** the <0.01 Hz DC baseline of raw CLE/CRE/CH via decimate-to-1 Hz + rolling median, plotted centered on each channel's session median with IQR-robust y-limits (sensor-dropout spikes clip off-screen; DC span reported in each title). (3) **Frequency characteristics:** 12-session-average Welch PSD per channel, per-session spectral centroid per band, and per-session band-power fractions. Loads each session once, caches lightweight per-session summaries to `signal_characterization_cache.pkl` (pass `--recompute` to rebuild) so figures re-tune without reloading raw data. Outputs `fig8_simplified_snr.png`, `fig9_mean_capacitance_drift.png`, `fig10_frequency_characteristics.png`, `signal_characterization_summary.csv`. Run: `.venv/Scripts/python.exe writeup/figures/signal_validation/signal_characterization.py`.

## 2026-07-09

- **Changed** `analysis/slow_wave/cap_swa_trial_examples.py` — per-session saved trial plots now include the RAW CAP + PSG waveforms at high resolution. For each example trial, in addition to the compact epoch-level overview (`examples/<sess>.png`), writes a full-width high-res figure `examples/<sess>_trial<NNN>.png` (dpi 150, 18in wide) stacking: stage strip + C1/C3 criteria (epoch) then raw 100 Hz signals — CAP CLE/CRE/CH (demeaned) and PSG EEG (0.3–40 Hz bandpassed for readability), ECG, Thorax, Flow, Pleth — over the trial ±5 min, trial shaded green with head-movement markers. Loads each raw session once via `load_session` (added `--n` for #examples/session and `--overview-only` to skip the raw figures). Shows the pre-onset head movement as large Thorax/Flow artifacts followed by regular breathing + quiet CAP inside the trial.
- **Added** `analysis/spindles/plot_spindle_windows_all.py` — writes ONE per-spindle window figure for every spindle in a session (single-column: EEG raw / EEG 11–16 / CLE-CRE,CLE,CRE,CH 11–16 / CAP spectrogram, ±3 s, span shaded; title = spindle #, stage, time, freq, dur) into a per-session subfolder `outputs/spindle_windows_<label>[_N2]/spindle_<NNNN>_<t>h.png`. `--n2-only` flag restricts to N2. Ran on S2N1 → 1,871 figures. Filters signals once per session; ~progress every 100.
- **Added** `analysis/spindles/spindle_ersp_control.py` + `plot_ersp_control.py` — controls for the CAP 0–3 Hz spindle bump. Reuses `session_ersp`; adds a `Classification Arousal` loader (same wall-clock alignment as spindles). Computes core-vs-baseline spectra for 5 conditions (spindle / random-N2 null / N2 arousals / spindle-with-arousal±5s / spindle-without-arousal) on CH + CLE-CRE. Result: random-N2 flat (CH +0.06 dB) and arousal-free spindles still bump (CH +0.43) → the bump is a genuine spindle-locked mechanical signature, not an artifact or arousal confound. Outputs `fig_spindle_ersp_control.png`, `spindle_ersp_control.npz/.csv`.
- **Changed** `analysis/spindles/plot_ersp.py` — corrected the two ERSP suptitles (were "no CAP transient at any frequency"; now state the validated 0–3 Hz spindle-locked bump).
- **Added** `scripts/analyze_k_biomarker_perwindow.py` — per-window k as a *secondary* biomarker (cache-only, `artifacts/mask_phase_a.parquet`; no raw reprocessing). Reproduces the rate confound of the supervised `k_gt = peaks_loose/GT` (within-session corr with rate = −0.845 cardiac / −0.561 resp), then defines a GT-free deployable marker `M = peaks_loose/hilbert` ("morphological multiplicity") that is rate-flat (within-session corr ≈ +0.08 card / +0.03 resp) and its rate-residual `k_resid`. Tests M with within-session/LOSO statistics only: lag-1 autocorr vs 200× epoch-shuffle null, within-session Kruskal-Wallis + Friedman direction-consistency (contrasted against the inflated pooled p), night1-vs-night2 ICC(1), and LOSO one-vs-rest logistic AUC (N3, REM). Outputs `reports/rates/mask/k_biomarker_perwindow.csv` (18,638-row marker table for downstream joins), `reports/rates/mask/k_biomarker_stats.csv`, `writeup/figures/mask_rate_detection/fig_k_biomarker.png`. NOT wired into the rate pipeline — Fig 9 keeps static per-session k. Run: `C:/Users/adity/anaconda3/python.exe scripts/analyze_k_biomarker_perwindow.py`.
- **Added** `scripts/cap_swa_trial_viewer.py` — interactive Plotly Dash browser for the CAP-SWA trials (companion to `session_viewer.py`, runs on port 8051). Driven entirely by `reports/slow_wave/cap_swa/trials/{trial_epochs.parquet, trials.csv}` so it starts instantly. Whole-night overview per session (hypnogram + CLE/CRE/CH means + C1/C3 hold criteria + HR + EEG delta) with every trial shaded green and a clickable onset marker; clicking a marker (or picking from the trial dropdown) pops that trial's pre/trial/post detail (6-panel, ±5 min, head-movement markers) into a panel below while the whole-night stays visible. Auto-saves the whole-night PNG on each session load; "Export ALL" button and `--export-all` CLI write every session's whole-night PNG + all 59 trial-window PNGs (via matplotlib — kaleido not installed). Outputs -> `reports/slow_wave/cap_swa/trials/viewer_exports/{<sess>_wholenight.png, <sess>/trial_NNN.png}`. Run: `.venv/Scripts/python.exe scripts/cap_swa_trial_viewer.py` then open http://localhost:8051.
- **Added** `analysis/spindles/spindle_ersp.py` + `plot_ersp.py` — spindle-triggered ERSP (baseline-corrected time-frequency, 0–45 Hz, ±8 s) for EEG + all CAP channels; hypothesis-free "does any band change vs its own baseline". EEG shows sigma; CAP shows no sigma but a small 0–3 Hz bump (see ANALYSIS_LOG). `session_ersp` bugfix: separate `faxis` variable so the returned freq axis is the filtered 58-bin one (was returning the unfiltered 65-bin axis → per-session IndexError). Outputs `fig_spindle_ersp_maps.png`, `fig_spindle_ersp_spectra.png`, `spindle_ersp.npz`, `spindle_ersp.csv`.
- **Added** `analysis/spindles/plot_spindle_windows.py` — single-event spindle windows: for the N strongest N2 spindles in a session, plots EEG raw + EEG 11–16 Hz + CAP 11–16 Hz for every channel (CLE−CRE, CLE, CRE, CH) + a CAP sigma spectrogram over a shared ±3 s window with the scored spindle span shaded. Shows event-level (not just averaged) that no CAP channel carries sigma. Output `outputs/fig_spindle_windows_<label>.png`.
- **Added** `analysis/spindles/spindle_indirect.py` — two follow-up tests over all 12 sessions, all CAP channels. #3 heart-rate route: spindle-triggered instantaneous HR from ECG R-peaks (positive control) vs CAP cardiac-band pulse peaks for each of CLE−CRE/CLE/CRE/CH, ±15 s, with a random-N2 null. #2 coherence route: whole-night magnitude-squared coherence EEG↔{CLE−CRE,CLE,CRE,CH} (leakage test) and CLE↔CRE (same-source anchor). Writes `spindle_hr_route.csv`, `spindle_coherence.csv`, `spindle_hr_triggered.npz`, `spindle_coherence.npz` (per-channel columns/keys).
- **Added** `analysis/spindles/plot_indirect.py` — figures for the two routes, all channels: `fig_spindle_hr_triggered.png` (2×3 — ECG biphasic ±~1.4 bpm spindle-cardiac beat, every CAP channel flat vs null) and `fig_spindle_coherence.png` (all EEG-CAP pairs ≈0 incl. sigma; CLE-CRE anchor peaks at resp/cardiac, proving the estimator works).
- **Changed** `analysis/slow_wave/cap_swa_trials.py` — REDEFINED trials to the user's exact 3-point criteria: C1 slow mean drift of a SINGLE capacitive channel (CLE/CRE/CH, not the CLE−CRE difference; "or" = min |slope| across channels), C2 trial INITIATED BY A HEAD MOVEMENT (accel event ≤3 epochs before onset), C3 low-variance/slow thorax. Dropped the prior quiescence-throughout hold condition — the accelerometer now supplies the movement trigger. Recomputes single-channel DC means + movements from raw sessions; physiology merged from the aligned cap_swa parquet (100% stage agreement). q=0.5 → 59 movement-initiated trials (was 91): more selective, more N3-enriched (2.15× vs 1.71×), stronger bradycardia (HR ↓ 6/6, −2.2 BPM vs −1.3). Outputs overwrite the prior trials/ files.
- **Changed** `analysis/slow_wave/cap_swa_trial_examples.py` — example panels updated for the new criteria: single-channel CLE/CRE/CH means (demeaned) replace the CLE−CRE mean, hold criteria show C1/C3 sub-scores, and head-movement events are marked (orange), highlighting the initiating movement before each trial onset.
- **Added** `analysis/slow_wave/cap_swa_trial_examples.py` — per-session example-trial figures with pre/post context. For each session picks a few representative trials (ranked by N3 content then duration) and plots each with a 5-min pad before/after: stage strip, the three criteria sub-scores (+ q line), CLE−CRE mean (D1), HR, EEG delta, thorax+accel. Trial span shaded, onset/offset marked. Shows the trials are bounded by movement/DC-step events (thorax+accel spike and a CLE−CRE mean step at offset) with flat DC + quiescence + elevated EEG delta inside. Outputs `reports/slow_wave/cap_swa/trials/examples/<session>.png` (12 sessions).
- **Added** `analysis/slow_wave/cap_swa_trials.py` — TRIAL-based (observational) analysis of the CAP-SWA definition, per user request: instead of the graded geometric-mean score fed to a classifier, find discrete windows where the three mechanical criteria (D1 slow CLE−CRE DC slope, D3 slow thorax amplitude, Dq low accel RMS) ALL hold as a conjunction — each per-session sub-score ≥ q — in sustained runs of ≥4 epochs, treat each run as a trial, and characterize the trials. Reads the existing `all_epoch_features.parquet` (no re-extraction). **Findings (q=0.5): 91 trials, median 3.0 min (max 10.5), 15.5% of the night, 11–19 per subject.** Trials are predominantly consolidated NREM, NOT N3-specific: 67% of trial epochs are N2, 15% N3, 13% N1 (95% NREM); N3 enriched 1.7× over base rate, Wake depleted to 0.31×, REM to 0.68×. Stricter conjunction purifies toward N3 (q=0.7 → 2.8× N3 enrichment but only 9 trials covering 9% of N3). Physiology during trials reproduces the graded-score signature (per-subject direction): HR ↓ 6/6 (−1.3 BPM, bradycardia), EEG delta ↑ 5/6, thorax amplitude ↓ 4/6, CAP cardiac freq ↑ 6/6, PPG−CAP divergence reverses 6/6, RR flat. Onset-triggered averages show HR/accel/thorax dipping to a trough at trial onset while EEG delta climbs and stays elevated — entry into a deep, quiet, delta-rich state. Outputs: `reports/slow_wave/cap_swa/trials/{trials.csv, trial_epochs.parquet, stage_composition.csv, physiology_contrasts.csv, onset_triggered.csv, cap_swa_trials.png, per_session/<sess>.png}`. Manuscript not edited.
- **Added** `analysis/slow_wave/ladder_bands_spectrogram.py` — side-by-side band-isolated ladder view: CH + CRE spectrograms with respiratory-band rungs (cyan, Δf 0.12–0.5) and cardiac-band rungs (magenta, Δf 0.5–1.6) drawn on the members, plus a per-channel ladder-occupancy timeline. Supports `--zoom`. Visually confirms the channel dissociation (CH cardiac-dominant, CRE respiratory-dominant) and the intermittent, episode-clustered nature of ladders. Outputs `reports/slow_wave/ladder_quantify/ladder_bands_<session>[_zoom].png`.
- **Changed** `analysis/slow_wave/ladder_spectrogram.py` — `comb_fit` gains `df_lo`/`df_hi` (spacing band) and `max_min_k` (require a rung at k≤max_min_k so a banded spacing cannot be fit to only high harmonics of the other rhythm). `analysis/slow_wave/ladder_quantify.py` now runs combined + resp + cardiac modes independently per window/channel and writes `summary_by_band.csv`.
- **Added** `analysis/slow_wave/show_ladder_examples.py` — per-window waveform + fine PSD with detected ladder members and k·f0 guides, for eyeballing whether a detection is a real ladder. Surfaced that the integer-ratio member count over-counts (grabs non-contiguous high-order cardiac ridges as respiratory harmonics).
- **Added** `analysis/slow_wave/ladder_spectrogram.py` — CH spectrogram + persistent-ridge overlay + comb-fit ladder quantifier (single session viz). `comb_fit()`: ladder = equally-spaced prominent ridges; searches Δf, scores n_rungs × span-coverage, refines Δf by least squares, flags harmonic (comb through 0) vs inharmonic. Δf search 0.15–1.6 Hz, prominence filter ≥2× floor.
- **Added** `analysis/slow_wave/ladder_quantify.py` — multichannel batch quantifier (CH/CLE/CRE, all 12 sessions). Per-window per-channel comb ladders; a window carries a ladder if any channel detects ≥3 rungs. Band split (respiratory f0<0.6 vs cardiac), harmonic/inharmonic, per-stage prevalence, per-channel comparison.
- **Output** `reports/slow_wave/ladder_quantify/{per_window_channels.parquet, per_window_combined.parquet, summary.csv, channel_ladder_counts.csv}`; per-session `reports/slow_wave/harmonic_rigor/ladder_spectrogram_<session>.png`.
- **Key result** Clean prominent ladders are intermittent (2% of non-motion windows, up to 20% S6N2), correcting the loose 85%. Channel dissociation: CH→cardiac ladder (f0~1.1 Hz), CRE→respiratory ladder (f0~0.3 Hz), CLE negligible. 76% cardiac / 24% respiratory; 63% harmonic / 37% inharmonic.

- **Added** `writeup/figures/cap_swa/make_cap_swa_figure.py` + `fig_cap_swa_definition.png` — publication figure for the CAP-SWA mechanical/autonomic definition (from cached cap_swa outputs, no recompute). Four panels: (A) per-subject swa_score→N3 AUC (6/6 above chance, mean 0.675), (B) HR non-SWA→CAP-SWA per subject (bradycardia), (C) professor-hypotheses table (predicted vs observed direction + per-subject consistency + verdict), (D) threshold sweep showing 0.60 at the F1 maximum.
- **Added** `writeup/edits/cap_swa_section_draft.md` — ready-to-paste manuscript Methods + Results (Part C) + Table + Discussion for the CAP-SWA definition. Honest-negative framing: mask does not measure cortical SWA (§3.5) or spindles (§3.6), but the mechanical/autonomic score marks N3 with consistent cross-subject direction (per-subject AUC 0.675, 6/6); five of six autonomic hypotheses contradicted (HR bradycardia not tachycardia, k-deviation shrinks, PPG/CAP divergence reverses, RR flat, movement does not precede onset), EEG delta up as a weak trend. Per-subject direction counts as evidence per OUTLINE rule 3. Manuscript docx not edited.

- **Added** `analysis/slow_wave/swa_classifier_experiment.py` — CAP-SWA classifier ablation + definition tuning (Workstream C, steps 1–2). Loads the self-consistent `reports/slow_wave/cap_swa/all_epoch_features.parquet`; PART A runs a LOSO N3 ablation (direct-score AUC + logistic + GBM) over nested feature sets (swa_score / swa_subscores / mechanical); PART B sweeps the swa_score threshold (0.40–0.85, sustained-bout vs raw) reporting precision/recall/F1; PART C sweeps the geometric-mean sub-score weighting over the simplex reporting pooled + per-subject N3 AUC. Adds a `direct_score_auc()` helper — a depth-3 GBM on one continuous feature ties its outputs and spuriously deflates AUC toward 0.5, so the composite score is reported directly (tie-free). **Key results:** per-subject `swa_score`→N3 AUC 0.675 ± 0.073 [0.575, 0.759], 6/6 subjects above chance (consistent cross-subject direction, unlike ridge/HER/mean-value); mechanical set 0.692 per-subject; CAP-only (slow-DC + quiescence, no PSG thorax belt) pooled 0.666 / per-subject 0.671 ± 0.042; the 0.60 threshold is F1-optimal (0.281); equal sub-score weighting is within 0.015 of the swept optimum. **Note:** deliberately does NOT fuse with the May-28 `sws_features.parquet` cache — that cache uses an older epoch/stage alignment (~43% stage-label disagreement) that misaligns the join and drove full_cap below chance; a valid fusion requires re-extraction (flagged in ANALYSIS_LOG). Manuscript not edited.
- **Output** `reports/slow_wave/cap_swa/classifier/{loso_ablation.csv, loso_ablation_folds.csv, threshold_sweep.csv, weighting_sweep.csv, classifier_experiment.png}`.

- **Added** `analysis/mean_value/mean_value_vs_stage.py` — professor Results-opener directive ("start with mean value / raw signal changes vs sleep stages"). Per-30s-epoch DC mean + <0.05 Hz VLF baseline for CLE/CRE/CLE-CRE across 12 sessions; per-session z-score, slow-trend-removed and motion-clean variants; Kruskal-Wallis per stage, per-subject direction-consistency check, LOSO AUC; spectrogram+hypnogram+mean-trace overlays. Finding: significant pooled association (KW p up to 1e-25, survives detrending/motion controls) but **subject-dependent in direction** (no unanimous contrast) with near-chance LOSO AUC (0.40–0.57) — not a universal biomarker; suitable only as within-subject descriptive context. Manuscript not edited.
- **Changed** `analysis/mean_value/mean_value_vs_stage.py` — extended mean-value analysis to **all channels + accelerometer**: added CLE+CRE average and acc_mag as mean-value channels (now CLE, CRE, CLE-CRE, CLE+CRE, acc); per-subject direction check and LOSO AUC now looped over every channel; per-channel boxplot + subject-direction heatmap figures. Confirms/strengthens the original verdict: every channel is significant when pooled (KW p 1e-4 → 1e-31) but **all stage contrasts are subject-dependent** across all 5 channels (only exception: CLE+CRE Wake-vs-sleep, 6/6 subjects lower, yet still near-chance LOSO AUC 0.55); LOSO AUC across all channels/contrasts spans 0.29–0.61 (chance). acc mean tracks head posture, not stage (Wake not highest). Stage ordering flips between raw and detrended for CRE/CLE+CRE, confirming overnight drift dominates.
- **Output** `reports/mean_value/{mean_value_epochs.csv, mean_value_stage_stats.csv, mean_value_subject_direction.csv, mean_value_loso_auc.csv}`; `notebooks/plots/mean_value/{spectrogram_hypno_*, meanvalue_hypno_*, boxplot_<ch>[_detr], subject_direction_<ch>}.png` for ch ∈ {CLE, CRE, CLE_CRE, CLEplusCRE, acc}.
- **Added** `analysis/mean_value/raw_stats_vs_stage.py` — extends the mean-value study to a spanning set of time-domain per-window raw statistics vs stage, all 5 channels: level (mean), dispersion (std, iqr), trend (slope_win within-epoch, slope_vlf <0.05 Hz baseline slope), roughness (linelen), complexity (Hjorth mobility, complexity), shape (skew, kurt). Per (stat×channel): Kruskal-Wallis, per-subject direction consistency, LOSO AUC; overview figure = LOSO-AUC grid (stat×channel) per contrast. **Key result — first cross-subject-consistent CAP stage signal:** dispersion + complexity statistics separate **Wake vs sleep** robustly and 6/6-consistently (std/mobility/complexity LOSO AUC 0.72–0.77 on CAP channels, 0.65–0.71 on acc) — an arousal/movement detector that generalizes across subjects, unlike the DC mean (0.55, subject-dependent). N3-vs-rest is weak-but-consistent for Hjorth mobility/complexity (AUC 0.60–0.66, 6/6). REM and the slope/skew/kurt families stay near chance; slope medians do not separate stages (only their Wake-time variability, already captured by std). Manuscript not edited.
- **Output** `reports/mean_value/{raw_stats_epochs.csv, raw_stats_kw.csv, raw_stats_subject_direction.csv, raw_stats_loso_auc.csv}`; `notebooks/plots/mean_value/{loso_grid_<contrast>, boxplot_{std,slope_vlf}_CLE_CRE, subject_direction_{std,slope_vlf}_CLE_CRE}.png`.
- **Changed** `analysis/mean_value/mean_value_vs_stage.py` — generic boxplot title ("Per-window {feat} by stage") so the reused figure helper reads correctly for non-mean statistics.

- **Changed** `writeup/main/CAP_sleep_mask_manuscript_main.docx` — addressed professor review markup. (1) Fixed section numbering: Results subsections renumbered 3.1–3.5 → 4.1–4.5 (were duplicating Methods), Discussion 4→5, Limitations 5→6, Conclusion 6→7 (headings only; no cross-refs affected — in-text fields are EndNote citations). (2) Removed remaining informal "battery" jargon in prose ("extensive battery"→"suite"; "Tracking battery:"→"Tracking evaluation:"; figure caption "(tracking battery)"→"(tracking evaluation)"), preserving the `symmetric_tracking_battery.csv` filename. Still open: professor's directive to open Results with a raw-signal mean-value vs sleep-stage analysis (analysis in progress). Untouched original backed up in session scratchpad.
- **Added** `analysis/slow_wave/harmonic_rigor.py` — harmonic rigor (Workstream B). Shared `best_ladder()` scorer; per-session per-k surrogate null (random ridge frequencies) with 95th-pct threshold; strong-ladder criterion (≥3 members, ratio_tol 0.06) that beats the null; calibrated confidence (ratio_quality × amplitude-decay-monotonicity); non-ladder window classification (motion/quiet/single-tone/broadband/multi-non-harmonic) cross-tabbed against sleep stage; survival + confidence + otherwise-by-stage summary figure.
- **Output** `reports/slow_wave/harmonic_rigor/{ladder_windows.parquet, null_summary.csv, otherwise_crosstab.csv, harmonic_rigor.png}`.

- **Changed** `sleep_monitor/harmonics.py` — `detect_persistent_ridges()` now stashes the pre-smoothing peak trace (`freq_trace_raw`) and computes per-ridge flatness/consistency metrics (`freq_std`, `freq_cv`, `drift_slope`, `coverage`, `flatness`) in a new Step 5e. Additive keys only; existing consumers unaffected (test suite: 90 pass, 1 pre-existing unrelated preprocessing failure).
- **Added** `analysis/slow_wave/ridge_consolidation.py` — ridge consolidation (Workstream A). Baseline-vs-flat-favoring config comparison (flat-ridge yield + cross-session reproducibility); per-ridge table with flatness + dominant stage; per-window ridge features vs ALL five stages (KW + one-vs-rest AUC + per-subject direction); flatness-by-stage + AUC-heatmap summary figure.
- **Output** `reports/slow_wave/ridge_consolidation/{per_ridge.csv, per_epoch_features.parquet, stage_association.csv, retune_comparison.csv, ridge_consolidation.png}`.

- **Added** `analysis/slow_wave/cap_swa_definition.py` — CAP-SWA operational definition (Workstream C). Per-epoch mechanical/physiology/validation feature extraction; graded SWA score (percentile geometric mean of slow-DC + slow-thorax + quiescence sub-scores); binary sustained-bout candidate label; movement-initiation precursor test with matched-random null (H2); per-subject SWA-vs-non-SWA contrast for the professor's autonomic hypotheses (H4 HR, H5 RR, H6 CAP/thorax k-deviation, H7 PPG-vs-CAP cardiac freq, H8 EEG delta) with Bonferroni; SWA-score N3 AUC.
- **Output** `reports/slow_wave/cap_swa/<session>/epoch_features.csv` + `night_overview.png` (12 sessions), `all_epoch_features.parquet`, `hypothesis_summary.csv`, `movement_initiation.csv`.

## 2026-06-18

- **Added** `scripts/generate_manuscript_docx.py` — generates complete manuscript draft (Methods, Results, Discussion, Limitations, Conclusion + Open Items) as Word document. Embeds 14 key figures, 6 tables, reconciled against all post-consensus source CSVs. Title/Abstract/Intro are placeholders.
- **Output** `writeup/CAP_sleep_mask_manuscript.docx` — 18 MB, 14 embedded figures, 6 data tables, OPEN-ITEMS section listing revised claims and unverified numbers
- **Updated** `writeup/paper/KEY_NUMBERS.md` — reconciled all numbers against post-consensus GT, tracking-FAIL finding, LOSO harmonics. Deprecated pre-consensus k-biomarker and rate_consolidation numbers.
- **Updated** `writeup/paper/CLAIMS.md` — reconciled 28 claims: 7 revised/deprecated (k-biomarker, rate pipeline numbers, multi-channel), 4 added (tracking FAIL, two operating points, ceiling, LOSO N3). Revision log appended.

- **Added** `scripts/evaluate_symmetric_tracking.py` — symmetric resp+cardiac tracking evaluation from cached Phase A data. Detector B (peaks_loose + hilbert, 5-channel mean-fusion, k-calibrated, minimal smoothing) vs spectral baseline. Tracking battery: within-session r, delta-tracking, transient/steady split, 200-iteration temporal-shuffle null. Two operating points framing. Achievable ceiling (Flow vs RIPSum r=0.47).
- **Output** `writeup/figures/mask_rate_detection/fig18_mae_heatmap.png` — multichannel x multimethod MAE heatmap with per-session IQR, resp + cardiac side by side
- **Output** `writeup/figures/mask_rate_detection/fig19_tracking_r_bars.png` — per-session within-session r (DetB + spectral) with shuffle-null 5th-95th bands
- **Output** `writeup/figures/mask_rate_detection/fig20_delta_transient.png` — delta-tracking r + transient vs steady segment analysis
- **Output** `writeup/figures/mask_rate_detection/fig21_operating_points.png` — MAE vs tracking r tradeoff, per session and band
- **Output** `writeup/figures/mask_rate_detection/fig22_fullnight_traces.png` — full-night GT vs DetB vs spectral traces (4 sessions x 2 bands)
- **Output** `writeup/figures/mask_rate_detection/fig23_ceiling_comparison.png` — mask tracking r vs achievable ceiling (Flow vs RIPSum)
- **Output** `reports/rates/mask/symmetric_tracking_{mae_table,battery,ceiling}.csv` — detailed per-session results
- **Output** `artifacts/detB_{resp,card}.parquet` — Detector B fused rate estimates per epoch
- **Key results** RESP: FAIL (r=+0.058, p=0.34, 4/12 beat null). CARD: FAIL (r=-0.188, p=0.85, 3/12 beat null). Ceiling: Flow vs RIPSum r=+0.47. Mask recovers mean rate only.

- **Added** `analysis/slow_wave/paper_ridge_demo.py` — paper-ready harmonic ridge demo: (A) per-session spectrogram + ridge overlay with hypnogram, (B) pooled quantification (violin, heatmap, ROC, KW/MW-U), (C) Stage 4 LOSO N3 classifier (RF, 4 ridge features). Completes all 4 stages of the harmonic structure detection plan.
- **Output** `writeup/figures/harmonics/paper_overlay_*.png` — 12 per-session spectrogram + ridge overlay figures (CRE channel)
- **Output** `writeup/figures/harmonics/paper_quantification.png` — 6-panel pooled quantification figure
- **Output** `writeup/figures/harmonics/paper_n3_loso.png` — LOSO N3 classification results (ROC, metrics, feature importance)
- **Output** `reports/slow_wave/paper_n3_loso_metrics.csv` — per-fold LOSO metrics
- **Key result** Ridge features are statistically significant (KW p<1e-16) but near-chance N3 classifiers (LOSO AUC=0.534, mean F1=0.095). Subject-dependent direction confirmed.
- **Added** `writeup/figures/signal_validation/generate_band_energy.py` — generates fig5/6/7 demonstrating CAP mask energy in resp (0.1–0.5 Hz) and cardiac (0.5–3.0 Hz) bands. Uses `sleep_monitor/spectral.py` Welch PSD + `sleep_monitor/viz.py` for spectrograms. Three panels: annotated spectrograms (3 sessions), band-power time course vs PSG GT rate, in-band SNR summary (boxplots + time course + mean PSD).
- **Output** `writeup/figures/signal_validation/fig5_cap_spectrogram_bands.png` — CLE−CRE spectrograms (0–5 Hz) with resp/cardiac band overlays
- **Output** `writeup/figures/signal_validation/fig6_bandpower_vs_psg_rate.png` — sliding 60s band power vs PSG GT rate, dual-axis
- **Output** `writeup/figures/signal_validation/fig7_inband_snr.png` — in-band SNR boxplots, SNR time course, mean PSD with band annotations
- **Added** `sleep_monitor/ground_truth.py:gt_resp_rate_consensus()` — loads consolidated multi-signal resp GT from `artifacts/consolidated_resp_gt.parquet`, returns consensus rate on any time grid (exact sampling). Module-level cache for repeated calls.
- **Changed** `sleep_monitor/ground_truth.py:gt_sliding_rates()` — new `resp_method=` arg (default `'consensus'`). Uses multi-signal consensus for resp GT; falls back gracefully to Flow→Thorax peak detection when parquet missing or session absent (e.g. validation recordings).
- **Added** `scripts/reattach_consensus_gt.py` — re-attaches consensus resp GT to `artifacts/mask_phase_a.parquet` on the IDENTICAL grid (exact join on session+t_hr, no merge_asof). 46,595/46,595 resp rows matched (9,319 unique epochs × 5 channels), 295 previously-NaN epochs now filled. Cardiac untouched. Median |delta| = 0.06 br/min, 96.5% of epochs changed.

---

## 2026-06-17

- **Added** `scripts/analyze_adaptive_k_and_oracle.py` — cache-only follow-up analysis (no raw reprocessing): self-supervised adaptive k(t), per-epoch oracle headroom (channel/method/full), channel-win distribution. Outputs CSVs in `reports/rates/mask/` + figs 7-9.
- **Added** `CONTINUATION_RATE_DETECTION.md` — handoff doc for next CLI session (cache locations, findings, prioritized next steps)
- **Output** `writeup/figures/mask_rate_detection/fig{7,8,9}*.png` — oracle headroom, adaptive k, channel diversity
- **Key finding** Cardiac channel-diversity oracle = 1.58 BPM (vs fused 3.91) — large untapped headroom; resp headroom is method-diversity not channel (oracle-method 0.54). Self-sup adaptive k fails for cardiac (no good anchor).
- **Added** `scripts/run_mask_rate_detection.py` — paper-ready mask rate detection pipeline (6 phases: raw rates, Smart Fusion + multi-channel SQI, k-calibration + smoothing, evaluation, failure analysis, multi-channel value). Checkpointed at each phase (parquet). Generates 9 paper figures + per-session CSV.
- **Output** `writeup/figures/mask_rate_detection/` — 9 figures (pipeline progression, Bland-Altman, per-stage MAE, time series best/worst × 2 bands, failure analysis, multi-channel value)
- **Output** `reports/rates/mask/` — per-session CSV, final_summary.json, pipeline log
- **Output** `artifacts/mask_phase_{a,b,c}.parquet` — cached intermediate results (93k, 18k, 112k rows)
- **Key results** Resp: MAE=1.09 br/min, bias=-0.3, LoA=[-4.7, 4.2]. Card: MAE=3.91 BPM, bias=-0.6, LoA=[-24.1, 22.9]. Best resp: diff/spectral (k≈0.97). Best card: multi-ch agreement fusion of peaks_loose (k≈1.95).
- **Added** `writeup/paper/` — manuscript scaffold with OUTLINE.md, CLAIMS.md (28 claims with evidence chains), FIGURES.md (25 main + 48 supplementary), TABLES.md (10 tables), KEY_NUMBERS.md, DRAFT.md
- **Added** `writeup/shared/SHARED_METHODS.md` — shared methods text (participants, preprocessing, ground truth)
- **Added** `writeup/README.md` — writeup directory documentation with workflow instructions
- **Superseded** `writeup/PAPER_TASK.md` — monolithic task spec replaced by structured paper/ directory (file kept for reference)

---

## 2026-06-11 (cont.)

### Best-of-Both Rate Pipeline + Updated Documentation
- **Added** `scripts/evaluate_best_pipeline.py` — unified best pipeline: Kalman (reactive, R x0.3, Q x2.0) for resp, hilbert for cardiac, multi-channel quality-weighted fusion, configurable median temporal smoothing (SMOOTH_WIN=7), per-session + LOSO k-calibration
- **Output** `reports/rates/best_pipeline/` — 12 time-series PNGs, 12 Bland-Altman PNGs, aggregate + per-stage plots, 2 CSVs
- Resp: **1.49 br/min** (per-session k), 1.95 (LOSO) — 42% over peaks/k baseline
- Cardiac: **4.11 BPM** (per-session k), 5.41 (LOSO) — 15% over single-channel hilbert/k
- **Updated** `scripts/generate_rate_consolidation_docx.py` — now two-part document: Part 1 (original consolidation) + Part 2 (hybrid pipeline with all 12 session time-series, Bland-Altman, aggregate stats, per-stage, 4 tables)
- **Output** `writeup/CAP_rate_consolidation_section.docx` — 35 figures, 4 tables

### SWA Validation — Lucey et al. 2019 Replication (Steps 0-4)
- **Added** `analysis/swa_validation/swa_pipeline.py` — shared spectral pipeline: FIR bandpass (0.5-40 Hz via `firwin`), 6-sec epoch Welch PSD, band powers (1-4.5 Hz total, sub-bands, 20-30 Hz EMG), relative power normalization, artifact rejection (EMG 97.5th pct + accelerometer)
- **Added** `analysis/swa_validation/run_swa_validation.py` — full pipeline runner: Steps 1-2 (process all 12 sessions, both EEG and CAP), Step 3 (Pearson/Spearman correlation, Bland-Altman, coherence, ROC/AUC for N3 detection), Step 4 (per-subject summary, 5 publication plots)
- **Added** `analysis/swa_validation/step0_inventory.py` — data inventory scanner
- **Fixed** `swa_pipeline.py:bandpass_fir()` — replaced `firls` with `firwin`: `firls` with narrow 0.1 Hz transition band at 0.5 Hz produced catastrophically ill-conditioned coefficients (range ±2794, signal amplification ×26M). `firwin` window method is numerically stable (coefficients ±0.16 to 0.79)
- **Fixed** `sleep_monitor/loader.py:load_session()` — removed `unit='ms', utc=True` from `pd.to_datetime()` call (was causing `time_start` to always be None for datetime strings)
- **Fixed** `sleep_monitor/loader.py:load_sleep_profile()` — complete rewrite: now parses wall-clock timestamps from Sleep Profile epochs and computes offset from CSV `timeSM` start time, handles midnight crossing, drops out-of-range epochs. Previously assumed epoch 0 = CSV time 0, causing up to 38.5 min misalignment
- **Output** `analysis/swa_validation/outputs/` — `swa_validation_results.csv`, `swa_validation_per_subject.csv`, 5 PNGs (swa_overlay, bland_altman, coherence, roc_curves, correlation_scatter)
- **Result**: Negative — CAP temple differential does not measure cortical SWA (r≈0, coherence≈0, N3 AUC≈0.5). EEG sanity check confirms pipeline correctness (AUC=0.740).

### Hybrid Rate Pipeline — Phase 4: Streaming Demo
- **Added** `scripts/demo_realtime_rates.py` — real-time streaming rate tracker demo: `KalmanState` class for lightweight scalar Kalman filter, epoch-by-epoch processing with spectral + adaptive_peaks → Kalman fusion, live console display, summary plot
- **Output** `reports/rates/hybrid_phase4/streaming_demo_S1N1.png` — full-night time series (Kalman vs GT)
- S1N1: 954 epochs in 1.8s (16,348x real-time), resp MAE 1.88 br/min, cardiac MAE 21.29 BPM

### Hybrid Rate Pipeline — Phase 2: Multi-Channel Fusion
- **Added** `scripts/evaluate_multichannel.py` — runs spectral + adaptive_peaks → Kalman on 5 channels (CLE, CRE, CH, avg, diff) independently, quality-weighted fusion across channels, oracle (best per-window)
- **Output** `reports/rates/hybrid_phase2/` — 28 PNGs (per-session channel comparison, aggregate bars, heatmaps), 2 CSVs
- Resp: multi-ch 1.82 br/min (vs single-best 1.90, oracle 1.21) — 4% improvement
- Cardiac: multi-ch 17.74 BPM (vs single-best 21.22, oracle 8.63) — 16% improvement
- Oracle headroom shows substantial gains possible with better channel selection

### Hybrid Rate Pipeline — Phase 3: Formal Evaluation
- **Added** `scripts/evaluate_hybrid_pipeline.py` — full evaluation: per-session k-calibration for baseline (hilbert/k, peaks/k) and Kalman pipeline, LOSO k cross-validation, per-stage MAE, Bland-Altman, Wilcoxon tests
- **Output** `reports/rates/hybrid_phase3/` — 24 time-series PNGs, 3 Bland-Altman aggregate PNGs, 2 per-stage PNGs, 2 session-comparison PNGs, 3 CSVs (results, windows, k-calibration)

### Ridge Overlay v2 + Prominence Scoring
- **Updated** `sleep_monitor/harmonics.py`:
  - Step 5c: median-filter smoothing of ridge frequency traces (size=7)
  - Step 5d: per-ridge prominence traces (amp / local spectral floor ±0.3 Hz), median-filter smoothed
  - Added `compute_prominence_score()` — per-window max ridge prominence (gated at min_prominence=2.0), temporally smoothed (median filter size=15 ≈ 3.75 min), normalized to [0,1]
- **Updated** `analysis/slow_wave/run_ridge_overlay.py`:
  - `MIN_PERSIST_SEC` 180→300 (5 min minimum ridge)
  - Added `compute_fine_spectrogram()` — high-res visual background (nperseg=2048, nfft=4096)
  - Replaced harmonic-ladder scoring with prominence-based scoring throughout
  - Ridges colored by prominence (SNR vs floor) instead of raw amplitude; labels show "freq (Nx)"
  - Removed ladder dots, pick_best_channel, plot_multichannel_comparison
  - 6-row stacked layout, motion as red semi-transparent overlay, output to `reports/slow_wave/overlay/`
  - Figure size 22x20 at 200 DPI
- **Deleted** old superseded outputs: `harmonics_*.png`, `harmonic_ladders_*.png/.parquet`, `ridge_multichannel_*.png` from `reports/slow_wave/`
- **Added** `writeup/harmonics/generate_harmonics_docx.py` — Methods + Results section for harmonic ridge prominence analysis
- **Added** `writeup/harmonics/CAP_harmonic_ridge_analysis.docx` — generated Word document with 3 figures + Table 1
- **Added** `writeup/figures/harmonics/` — key figures (S1N1, S4N2 overlays, score-by-stage boxplot)

### Hybrid Rate Pipeline — Phase 1: Kalman Rate Tracker
- **Added** `kalman_rate_track()` in `sleep_monitor/rates.py` — scalar Kalman filter fusing spectral + adaptive_peaks per-window estimates with physiological rate-of-change constraints. Auto-selects Q from band (resp: 2 br/min/epoch, cardiac: 5 BPM/epoch). Handles NaN gaps, clamps to band bounds.
- **Updated** `sleep_monitor/__init__.py` — exported `kalman_rate_track`
- **Added** `tests/test_rates.py::TestKalmanRateTrack` — 6 tests (constant signal, noise smoothing, NaN gaps, all-NaN, length, bounds)
- **Added** `scripts/benchmark_kalman_tracker.py` — full benchmark: time-series, Bland-Altman, per-stage, aggregate bar chart, improvement heatmap
- **Output** `reports/rates/hybrid_phase1/` — 24 time-series PNGs, 24 Bland-Altman PNGs, 4 aggregate plots, 2 CSVs

### Hybrid Rate Pipeline — Phase 0: Adaptive Peak Detector
- **Added** `rate_adaptive_peaks()` in `sleep_monitor/rates.py` — spectral-guided, amplitude-adaptive peak detector with IPI validation. Uses spectral peak for min_distance, rolling MAD for prominence, and inter-peak-interval CV check with MAD-based outlier rejection.
- **Updated** `sleep_monitor/config.py` — added `adaptive_peaks` to METHOD_NAMES, METHOD_LABELS, METHOD_COLORS
- **Updated** `sleep_monitor/__init__.py` — exported `rate_adaptive_peaks`
- **Updated** `estimate_rate()` — now includes `adaptive_peaks` in output dict (6 methods)
- **Added** `tests/test_rates.py::TestRateAdaptivePeaks` — 7 tests (pure sine, noise robustness, short signal, erratic peaks, integration, amplitude drift)
- **Added** `scripts/benchmark_adaptive_peaks.py` — benchmark script comparing all methods on 12 sessions

---

## 2026-06-11

### SWA Validation
- **Added** `analysis/swa_validation/CLAUDE.md` — SWA validation workspace: Lucey et al. 2019 replication plan, Steps 0-4, deliverables, working rules
- **Added** `analysis/swa_validation/step0_inventory.py` — Step 0 data inventory script: scans all 12 sessions, reports format/channels/rates/alignment/quality

### Loader — staging alignment fix
- **Fixed** `sleep_monitor/loader.py` `load_session()` — `time_start` was always None due to `pd.to_datetime(val, unit='ms')` on datetime strings; removed `unit='ms'`
- **Fixed** `sleep_monitor/loader.py` `load_sleep_profile()` — now parses wall-clock timestamps from Sleep Profile epoch lines, aligns to CSV time via `session.time_start`. Drops epochs outside CSV window. Handles midnight crossing. Previously assigned epoch 0 to t=0 regardless of PSG→CSV offset (up to 38.5 min for S1-S2 sessions).

### Slow Wave / Harmonic Detection
- **Updated** `sleep_monitor/harmonics.py` — added `detect_persistent_ridges()`: temporally-continuous ridge tracking with motion masking, fragment merging, harmonic group detection, and continuous harmonic strength scoring
- **Added** `analysis/slow_wave/detect_sws.py` — LOSO N3 binary classifier from CAP features (motion, band power, spectral entropy, rate regularity, DC stability, coherence, harmonic features)
- **Added** `analysis/slow_wave/detect_trials.py` — trial-based SWS exploration: finds sleep segments matching physiological criteria (DC slope, post-movement settling, thorax smoothness)
- **Added** `analysis/slow_wave/plot_settling_events.py` — post-movement settling event visualization: ±15 min windows with stacked panels (accel, CAP, thorax, cardiac/resp rates, PPG, stage)
- **Added** `analysis/slow_wave/plot_trial_signals.py` — full-night raw CAP + PSG time series with detected trial regions highlighted (9 panels + spectrogram)
- **Added** `analysis/slow_wave/run_harmonic_ladders.py` — harmonic ladder detection via concurrent persistent ridges with integer-ratio grouping, prevalence by sleep stage
- **Added** `analysis/slow_wave/run_ridge_overlay.py` — consolidated harmonic ridge overlay: artifact removal, persistent ridges, continuous harmonic score, rich 4-row overlay plots, per-epoch parquet. Run on all 12 sessions; CRE dominant in 9/12.
- **Added** `analysis/slow_wave/NEXT_RIDGE_OVERLAY.md` — pickup spec for v2: high-res spectrogram, 5-min min ridge, median-filtered flat traces, 3-channel stacked layout
- **Added** `analysis/slow_wave/verify_harmonics_overlay.py` — harmonic detection verification: 5 figure types overlaying detected peaks/ridges on spectrograms and PSDs

### Rate Consolidation
- **Added** `scripts/run_rate_consolidation.py` — multi-channel fused rate pipeline (6 phases): method benchmark, channel confidence fusion, CWT ridge tracker, Viterbi smoothing, combined evaluation, k-calibration. Fixed stage assignment and encoding bugs; all 6 phases run to completion. 23 figures + CSV to `writeup/figures/rate_consolidation/` and `reports/rates/`
- **Updated** `writeup/PAPER_TASK.md` — added Rate Consolidation section (done) with figure inventory and key numbers
- **Added** `scripts/generate_rate_consolidation_docx.py` — Word document generation for rate consolidation section
- **Added** `writeup/CAP_rate_consolidation_section.docx` — standalone rate consolidation writeup

### Projections / Staging
- **Added** `scripts/run_clustering_phase3.py` — Phase 3 clustering: GMM (k=3,4,5) + DBSCAN on supervised UMAP embeddings for all 12 sessions
- **Added** `scripts/run_pooled_phase4.py` — Phase 4 pooled cross-session projections: subject-level z-score, pooled PCA/UMAP/t-SNE, GMM k=4, LOSO evaluation
- **Added** `scripts/run_supervised_validation.py` — supervised UMAP train/test validation: train fraction sweep (25/50/75%), GMM predict on held-out embeddings
- **Added** `scripts/run_supervised_validation_v2.py` — supervised UMAP validation v2: kNN + Random Forest in raw vs UMAP space, separating embedding value from classification ability

### Signal Validation & Paper
- **Added** `scripts/paper_signal_validation_figures.py` — paper-ready signal validation figures (4 figs + 1 table): waveform example, frequency agreement, coherence + surrogates, channel comparison
- **Added** `scripts/generate_paper_docx.py` — Word document generator for CAP sleep analysis paper (Signal Validation Methods + Results)
- **Added** `writeup/CAP_sleep_analysis_paper.docx` — main paper document (signal validation section)
- **Added** `writeup/PAPER_TASK.md` — paper writing task: scope, prerequisites, missing figures, data sources, key numbers
- **Added** `writeup/figures/` — paper-ready figures organized by topic: `signal_validation/`, `rate_consolidation/`, `rate_accuracy/`, `harmonics/`, `spectrograms/`, `supervised_umap/`, `gmm_clustering/`, `k_biomarker/`, `sfn_abstract/`

### Tests
- **Added** `tests/` — unit test suite: `test_filters.py`, `test_preprocessing.py`, `test_rates.py` with shared `conftest.py`

### Writeup
- **Added** `writeup/` — paper writeup directory with SFN abstract drafts (V4, V5, updated results v1/v2), XML export, figures

---

## 2026-05-28

- **Added** `analysis/slow_wave/run_ridge_stage3.py` — Stage 3: persistent ridge features vs sleep stage for all 12 sessions x 3 channels. Per-epoch feature extraction, KW and MW-U tests, per-subject analysis, 4 plot types.
- **Added** reports: `stage3_ridge_epochs.parquet`, `stage3_summary.csv`, `stage3_ridge_features_by_stage.png`, `stage3_ridge_features_per_subject.png`, `stage3_n3_vs_rest.png`, 12x `stage3_ridge_timeseries_*.png`

## 2026-05-22

- **Added** `analysis/thorax/CLAUDE.md` — thorax prediction analysis workspace: 6 phases of investigation, scripts/artifacts/plots inventory, conclusions and implications
- **Added** `analysis/thorax/ANALYSIS_LOG.md` — chronological log of all thorax analysis phases (correlation, prediction, CAP-only, ablation, residualization, slow-trends)
- **Updated** `CLAUDE.md` — added thorax to analysis areas listing

- **Added** `analysis/slow_wave/plot_harmonics_s1n1.py` — 3 harmonic visualization plots (full-night traces + hypnogram, stage boxplots, dominant channel breakdown)
- **Added** 3 plots in `notebooks/plots/harmonics/`: `harmonics_fullnight_s1n1.png`, `harmonics_stage_boxplots_s1n1.png`, `harmonics_dominant_channel_s1n1.png`
- **Added** `sleep_monitor/harmonics.py` — harmonic structure detection module (3 methods: HPS, cepstral, explicit f0+harmonics). Sliding-window `detect_harmonics()` and `detect_harmonics_multichannel()` returning per-window DataFrame with f0, n_harmonics, harmonic_energy_ratio, cepstral_prominence, harmonic_decay_rate, dominant_channel
- **Updated** `sleep_monitor/__init__.py` — registered `detect_harmonics`, `detect_harmonics_multichannel` in package exports
- **Added** `CLAUDE.md` — lean root context file (project identity, data paths, package API, workflow rules)
- **Added** `.claudeignore` — excludes artifacts, plots, notebooks, venv from context window
- **Added** `analysis/` workspace structure with scoped CLAUDE.md per analysis area:
  - `analysis/rates/CLAUDE.md` — rate estimation, k-biomarker, validation context
  - `analysis/slow_wave/CLAUDE.md` — SWS detection hypotheses and approach
  - `analysis/staging/CLAUDE.md` — sleep phase classification plan and feature spec
  - `analysis/projections/CLAUDE.md` — PCA, UMAP, t-SNE, DMD, delay embedding inventory
- **Added** `analysis/README.md` — index of analysis workspaces
- **Updated** `analysis/slow_wave/CLAUDE.md` — harmonic structure detection plan (HPS, cepstral, explicit f0+harmonics), 4-stage implementation, running trace spec

## 2026-05-14

- **Added** `scripts/thorax_residual_analysis.py` — motion-residualized thorax prediction (tests direct CAP→thorax coupling)
- **Added** `scripts/thorax_predictor_caponly.py` — CAP-only thorax resp RMS predictor (4 tiers, no thorax lags, no stage_code)
- **Added** `scripts/_ablation_quick.py` — feature group ablation (CAP signal vs accel vs context)
- **Added** `artifacts/thorax_caponly_epochs.parquet` — enhanced 58-col epoch features (spectral, rate, cross-channel, context)
- **Added** `artifacts/thorax_caponly_results.csv` — CAP-only predictor results (4 tiers × 2 eval modes × 12 sessions)
- **Added** 17 plots in `notebooks/plots/thorax_analysis/caponly_*.png`

- **Added** `LOGGING_POLICY.md` — logging discipline for all code/analysis changes
- **Added** `CHANGELOG.md` — this file
- **Updated** `notebooks/ANALYSIS_LOG.md` — retroactive entries for ~30 previously unlogged scripts/notebooks

## 2026-04-30

- **Added** `notebooks/10_projection_cap12.ipynb` — 12 CAP-only feature 3D projection (UMAP/t-SNE)
- **Changed** `sleep_monitor/config.py` — expanded `APNEA_CODES` with Flow event types
- **Changed** `sleep_monitor/loader.py` — `_parse_flow_file` replaces `_parse_effort_file` for apnea events

## 2026-04-30

- **Added** `notebooks/analysis_k_biomarker.py` — k(t) biomarker Phases 1+2
- **Added** `notebooks/analysis_k_biomarker_phase3.py` — k(t) correlation with PSG biomarkers

## 2026-04-22

- **Added** `sleep_monitor/ground_truth.py` — ECG R-peaks (Pan-Tompkins) + Flow peak detection via neurokit2
- **Changed** `scripts/compute_rates.py` — uses `gt_sliding_rates()`, records GT signal source in metrics
- **Changed** `sleep_monitor/morphology.py` — `compute_rate_divisor()` returns continuous float, default k=1.0
- **Archived** analysis scripts from `notebooks/` to `archive/rate_exploration/`

## 2026-04-16

- **Added** `sleep_monitor/rates.py` — `rate_hilbert_scaled_cardiac`, `rate_peaks_scaled_resp`, `calibrate_k_cardiac`, `calibrate_k_resp`
- **Added** `notebooks/analysis_hilbert_scaled_all_sessions.py` (now in `archive/rate_exploration/`)
- **Added** `notebooks/analysis_card_tuned_s1n1.py` (now in `archive/rate_exploration/`)
- **Added** `notebooks/demo_scaled_estimators.py` (now in `archive/rate_exploration/`)

## 2026-04-15

- **Added** `notebooks/analysis_default_rates_s1n1.py` (now in `archive/rate_exploration/`)
- **Added** `notebooks/analysis_resp_window_methods_s1n1.py` (now in `archive/rate_exploration/`)
- **Added** `notebooks/analysis_peak_ratio_sweep_s1n1.py` (now in `archive/rate_exploration/`)
- **Added** `notebooks/analysis_peak_ratio_all_sessions.py` (now in `archive/rate_exploration/`)

## 2026-04-12

- **Added** `notebooks/analysis_br_acf_peaks.py` (now in `archive/rate_exploration/`)

## Pre-2026-04-12 (undated, retroactive)

### Infrastructure scripts
- **Added** `scripts/cap_rates.py` — CLI for rate inspection (inspect/rates/metrics modes)
- **Added** `scripts/run_rate_detection.py` — default pipeline across all 12 sessions
- **Added** `scripts/compute_rates.py` — sliding-window rates with new GT
- **Added** `scripts/compute_eeg.py` — EEG band power by sleep stage
- **Added** `scripts/sweep.py` — grid search over (channel, preproc, estimator)
- **Added** `scripts/train_classifier.py` — rate-prediction classifier LOSO CV
- **Added** `scripts/add_psg.py` — one-time notebook code generator (can archive)

### Plotting scripts
- **Added** `scripts/plot_all_sessions_timeseries.py` — full-night + window time series
- **Added** `scripts/plot_best_rates.py` — best rate methods vs GT with stage overlay
- **Added** `scripts/plot_apnea_timeseries.py` — apnea event timelines per session
- **Added** `scripts/plot_apnea_fullnight.py` — 6-row full-night apnea overview

### Validation study scripts
- **Added** `scripts/run_validation.py` — per-epoch rate estimates vs GT, all 12 sessions
- **Added** `scripts/plot_validation.py` — Bland-Altman, scatter, stage boxplots
- **Added** `scripts/generate_validation_docs.py` — DOCX report generation

### Signal validation scripts
- **Added** `scripts/signal_validation.py` — per-epoch spectral/coherence/surrogate tests
- **Added** `scripts/merge_validation.py` — join signal validation with rate validation
- **Added** `scripts/plot_validation_report.py` — signal validation figures
- **Added** `scripts/signal_validation_enhanced.py` — 5 enhanced analyses + 5 channel combos
- **Added** `scripts/signal_validation_delay_pca.py` — delay-embedding PCA for channel combination
- **Added** `scripts/signal_validation_proof.py` — publication-quality signal validation proof
- **Added** `scripts/cardiac_coherence_test.py` — 16-condition factorial cardiac coherence test

### Rate accuracy scripts
- **Added** `scripts/rate_accuracy_analysis.py` — 4-channel rate accuracy, 30s epochs with annotations
- **Added** `scripts/rate_accuracy_docx.py` — DOCX report for rate accuracy

### ICP validation dataset scripts
- **Added** `scripts/load_validation.py` — loader for ICP validation dataset
- **Added** `scripts/validation_breath_rate.py` — breath-rate k per experiment mode
- **Added** `scripts/validation_peak_analysis.py` — phase-by-phase peak/rate analysis
- **Added** `scripts/validation_laydown_rates.py` — layDown-only respiratory/cardiac validation
- **Added** `scripts/plot_validation_rates.py` — best rates on ICP validation dataset

### Analysis notebooks (Python)
- **Added** `notebooks/analysis_morphology.py` — morphological cluster pipeline
- **Added** `notebooks/analysis_pca_stacked_cle_cre.py` — delay-embedded PCA on [CLE, CRE]
- **Added** `notebooks/analysis_delay_pca_cardiac.py` — delay-embedding PCA for cardiac
- **Added** `notebooks/analysis_dmd_cardiac.py` — DMD for cardiac rate
- **Added** `notebooks/analysis_dmd_rank_sweep.py` — DMD embedding dimension sweep
- **Added** `notebooks/analysis_sws_band_ratios.py` — SWS band power ratio analysis

### Jupyter notebooks
- **Added** `notebooks/01_overview.ipynb` — raw signal inspection, hypnograms
- **Added** `notebooks/02_rate_estimation.ipynb` — sliding-window rate visualization
- **Added** `notebooks/03_eeg_analysis.ipynb` — EEG spectrograms by stage
- **Added** `notebooks/04_metrics_summary.ipynb` — cross-subject accuracy tables
- **Added** `notebooks/05_method_search.ipynb` — best pipeline from sweep leaderboard
- **Added** `notebooks/06_classifier_results.ipynb` — rate classifier LOSO results
- **Added** `notebooks/07_validation_loader.ipynb` — ICP validation dataset loader
- **Added** `notebooks/08_cap_sleep_embedding.ipynb` — PCA/t-SNE/UMAP sleep embedding
- **Added** `notebooks/08_validation_loader.ipynb` — duplicate of 07 (validation loader)
- **Added** `notebooks/09_projection_3d.ipynb` — 3D UMAP/t-SNE with 40 features
