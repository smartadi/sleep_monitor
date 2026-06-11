# Ridge Overlay v2 — Pickup Spec

**Context**: `run_ridge_overlay.py` runs on all 12 sessions, produces per-session
4-row overlay plots + parquet. Current version works but needs visual + stability tuning.
Core module: `sleep_monitor/harmonics.py` (detect_persistent_ridges, compute_harmonic_score).

## Changes needed (4 items)

### 1. High-res spectrogram background

**Problem**: Currently plotting `psds_smooth` from the ridge detector (15s step, 0.1Hz bins).
Looks blocky/low-res.

**Fix in `plot_session()` and the new stacked plot function**:
- Compute a separate `scipy.signal.spectrogram` for the background image only
- Use: `nperseg=2048` (20s at 100Hz), `noverlap=1920` (~1.3s step), `nfft=4096` for zero-pad
- Gives ~0.025 Hz frequency resolution, ~1.3s time resolution
- Plot with `shading='gouraud'` for smooth interpolation
- Ridge detection stays on Welch PSDs — visual and detection are decoupled

```python
from scipy.signal import spectrogram as sp_spectrogram

def compute_fine_spectrogram(sig, fs=100.0, max_freq=5.0):
    f, t, Sxx = sp_spectrogram(sig, fs=fs, nperseg=2048, noverlap=1920,
                                nfft=4096, scaling='density')
    mask = f <= max_freq
    return t / 3600.0, f[mask], 10 * np.log10(Sxx[mask] + 1e-30)
```

### 2. Minimum 5-minute ridge duration

**Fix in `run_ridge_overlay.py`**:
```python
MIN_PERSIST_SEC = 300.0   # was 180.0
```

One-line change.

### 3. Flat (smooth) ridge traces

**Problem**: Ridge `freq_trace` follows the exact peak per window, causing jitter.

**Fix in `detect_persistent_ridges()` in `sleep_monitor/harmonics.py`**,
after Step 5b (merge fragments), before Step 6 (harmonic grouping):

```python
# ── Step 5c: Smooth ridge frequency traces ──
from scipy.ndimage import median_filter
for ridge in ridges:
    valid = ~np.isnan(ridge['freq_trace'])
    if valid.sum() < 7:
        continue
    # Extract valid segment, median-filter, put back
    freq_valid = ridge['freq_trace'][valid]
    freq_smooth = median_filter(freq_valid, size=7, mode='nearest')
    ridge['freq_trace'][valid] = freq_smooth
    ridge['median_freq'] = float(np.nanmedian(ridge['freq_trace']))
    ridge['label'] = f'{ridge["median_freq"]:.2f}Hz'
```

Add `from scipy.ndimage import median_filter` at top of file.

### 4. All 3 channels stacked (main plot)

**Problem**: Current overlay shows only best channel. User wants all 3 stacked.

**New layout** (replace current `plot_session`):
```
Row 0:  Hypnogram (height 0.4)
Row 1:  CH  spectrogram + ridges + harmonic events (height 1.8)
Row 2:  CLE spectrogram + ridges + harmonic events (height 1.8)
Row 3:  CRE spectrogram + ridges + harmonic events (height 1.8)
Row 4:  Harmonic score (all 3 channels overlaid, colored) (height 0.8)
Row 5:  Ridge stats (active count per channel, ladder f0) (height 0.8)
```

- Figure size: ~(22, 20) at 200 DPI
- Each spectrogram row: fine spectrogram background + ridge traces (top 20 labeled) + 
  green event bars + cyan ladder dots + motion mask red ticks
- Score row: 3 overlaid fill_between traces (CH blue, CLE green, CRE purple)
- Remove `pick_best_channel()` — no longer needed
- Remove `plot_multichannel_comparison()` — merged into main plot
- Keep `plot_score_by_stage()` as pooled summary

**Per-channel annotation** on each spectrogram row:
```
  CH: 135 ridges | 76 strong (4.7%) | median score 0.003
```

## Spectrogram gaps — CONFIRMED: motion masking, not data loss

Diagnostic run on S1N1 and S4N1 confirms:
- **S1N1**: no raw data gaps. Motion mask covers 14.7% of windows. One long block
  at 6.39–6.67 hr (69 windows = ~17 min) — this is the big black band.
- **S4N1**: no raw data gaps. Motion mask covers 18.1%. One long block at
  4.92–5.45 hr (126 windows = ~32 min).

The black bands are NaN PSDs from motion masking, rendered as black by inferno colormap.

**Fix**: When computing the fine spectrogram (scipy.signal.spectrogram), it uses raw
signal so motion windows will still have content — they'll show the motion artifact
instead of black. Overlay a semi-transparent red bar on motion-masked regions so
the user knows it's artifact, but the spectrogram is continuous. Remove the tiny red
ticks at top and use full-height red overlay at alpha=0.15 instead.

## Files to modify

1. `sleep_monitor/harmonics.py` — add median_filter import, add Step 5c (ridge smoothing)
2. `analysis/slow_wave/run_ridge_overlay.py` — MIN_PERSIST_SEC=300, replace plot_session
   with 6-row stacked layout, add compute_fine_spectrogram, remove pick_best_channel

## Test protocol

1. `pytest tests/ -v` before and after harmonics.py changes
2. Run on S1N1 only first: `python run_ridge_overlay.py S1N1`
3. Visually check: ridges should be smooth lines, spectrogram should be high-res,
   all 3 channels visible, no label clutter
4. Run all 12: `python run_ridge_overlay.py`
5. Check `reports/slow_wave/ridge_overlay_*.png` and `ridge_overlay_score_by_stage.png`

## Parquet schema (unchanged)

`reports/slow_wave/ridge_overlay_epochs.parquet` — one row per (session, channel, window):
session, subject, channel, t_hr, motion_masked, harmonic_score, ratio_quality,
n_ladder, ladder_f0, ladder_power, n_active_ridges, stage_code, stage_label
