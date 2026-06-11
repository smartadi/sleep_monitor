# Hybrid Real-Time Rate Pipeline — Multi-Channel

## Problem statement

Spectral (Welch PSD) gives the best single-method respiratory MAE (1.54 br/min no-k,
0.99 k-scaled) but it's a 30s window average. It cannot detect:
- Breath holds / apneas (rate drops to zero within window)
- Sighs / deep breaths (sudden rate changes)
- Arousal-related rate surges (ANS activation)
- Beat-to-beat HR variability (HRV features lost)

For a real-time deployment demo, we need event-level tracking. The goal is a **causal**
method (no future lookahead) that fuses spectral stability with instantaneous tracking.

## Evidence for complementarity (from Phase 1 analysis)

**Resp/diff channel, no k-scaling:**
- spectral MAE = 1.54 br/min (stable but blind to transients)
- hilbert MAE = 2.11 br/min (tracks instantaneous freq, noisier)
- peaks MAE = 2.72 br/min (event-level, overcounts)
- **spectral+hilbert oracle = 1.06 br/min** → 31% better than spectral alone
- **spectral+peaks oracle = 1.07 br/min** → same headroom
- hilbert beats spectral in **37.7%** of epochs (transient/changing-rate epochs)
- Error correlation: spectral-hilbert r=0.567 (moderate → they fail differently)
- ACF has lowest error correlation with spectral (r=0.131) but worse raw MAE

**Takeaway:** ~1.06 br/min is achievable if we can learn *when* to trust which method.
This is the "oracle" floor for a 2-method fusion.

## Architecture: Kalman-filter rate tracker

### Design principles
1. **Causal**: only uses past + current window data (no future lookahead)
2. **Dual timescale**: slow spectral prior + fast event-level updates
3. **Multi-channel**: independent per-channel trackers fused at output
4. **Adaptive confidence**: per-method, per-channel quality scores drive fusion weights
5. **Viterbi-compatible**: Kalman dynamics naturally enforce smooth transitions

### Per-channel pipeline (runs independently per channel)

```
Raw signal → Artifact removal (OLS) → Bandpass
                                          │
                    ┌─────────────────────┤
                    ▼                     ▼
              [30s Welch PSD]      [Sliding peak detector]
              spectral_rate        peak_times → IBI → inst_rate
              spectral_conf        peak_conf (prominence, regularity)
                    │                     │
                    └─────────┬───────────┘
                              ▼
                    [Kalman filter tracker]
                    state: [rate, rate_derivative]
                    prediction: rate(t) = rate(t-1) + rate_dot * dt
                    spectral update: every 30s, low noise
                    peak update: every detected event, higher noise
                              │
                              ▼
                    channel_rate(t), channel_confidence(t)
```

### Cross-channel fusion

```
CLE_rate, CLE_conf ──┐
CRE_rate, CRE_conf ──┤
CH_rate, CH_conf  ────┼──→ [Confidence-weighted fusion] ──→ fused_rate(t)
avg_rate, avg_conf ───┤         + agreement check
diff_rate, diff_conf ─┘
```

### Viterbi post-processing (optional, non-real-time)

For paper evaluation, apply Viterbi smoothing to the fused output
(max 2 br/min/epoch resp, 5 BPM/epoch cardiac). For real-time demo,
the Kalman filter already enforces smooth transitions via process noise.

## Implementation plan — 5 phases

### Phase 0: Adaptive peak detector (new module)

**Goal:** Replace the fixed-threshold peak detector with an adaptive one that uses
the spectral estimate as a prior for expected peak spacing.

**Module:** `sleep_monitor/peaks_adaptive.py`

**Algorithm:**
1. Bandpass signal → find all candidate peaks (scipy.signal.find_peaks, low prominence)
2. Score each candidate by:
   - Prominence relative to local RMS
   - Distance to expected IBI (from spectral prior or last accepted peak)
   - Amplitude consistency with recent peaks
3. Accept/reject candidates using an adaptive threshold that tightens when
   signal is clean (high SNR) and loosens when noisy
4. Output: peak_times, peak_prominences, IBI_series, instantaneous_rate

**Key parameters:**
- `expected_rate_hz`: from spectral estimate (or Kalman state)
- `tolerance`: how far from expected IBI before penalty (0.3 = ±30%)
- `min_prominence_factor`: minimum prominence as fraction of local RMS
- `max_gap_s`: maximum gap before declaring a "missed beat" (2× expected IBI)

**Test:** Compare detected peaks against GT (Flow/ECG) events on S1N1.
Count: true positives, false positives, false negatives. Target: F1 > 0.85 for resp.

### Phase 1: Kalman filter rate tracker (new module)

**Goal:** Causal rate estimator that fuses spectral and peak-based measurements.

**Module:** `sleep_monitor/rate_tracker.py`

**State vector:** `x = [rate_hz, rate_derivative_hz_per_sec]`

**Process model (prediction step):**
```
rate(t) = rate(t-1) + rate_dot(t-1) * dt
rate_dot(t) = rate_dot(t-1)   # constant rate-of-change assumption
Q = [[q_rate, 0], [0, q_dot]]  # process noise
```
- `q_rate`: allows rate to drift (tune: small for resp ~0.001, larger for cardiac ~0.01)
- `q_dot`: allows acceleration (tune: ~0.0001)

**Measurement model — spectral update (every 30s):**
```
z_spectral = spectral_rate_hz
R_spectral = f(spectral_confidence)  # low R when confident
```
- Spectral confidence from: SNR, spectral concentration, peak sharpness
- This is the "anchor" — pulls the tracker toward the dominant frequency

**Measurement model — peak update (every detected event):**
```
z_peak = 1.0 / IBI_seconds   # instantaneous rate from inter-peak interval
R_peak = f(peak_confidence, IBI_consistency)
```
- Peak confidence from: prominence, expected-IBI match, amplitude consistency
- High R (low trust) when IBI deviates far from expected → likely false peak
- Low R (high trust) when IBI is consistent with recent history

**Key feature — transient detection:**
When the Kalman innovation (z - H*x_predicted) exceeds a threshold:
- If from peaks: possible breath hold recovery, apnea, or sigh
- If from spectral: rate regime change (e.g., sleep stage transition)
- Flag epoch as "transient" for downstream analysis

**Output per timestep:**
- `rate_hz`: filtered rate estimate
- `rate_confidence`: inverse of state covariance
- `innovation`: deviation from prediction (transient indicator)
- `dominant_source`: which measurement drove the update ('spectral' or 'peak')

**Initialization:** First spectral estimate seeds the state. If no spectral available,
use the first 3 detected peaks.

### Phase 2: Multi-channel tracker

**Goal:** Run Phase 1 tracker on all 5 channels, fuse outputs.

**Module:** Extend `sleep_monitor/rate_tracker.py` with `MultiChannelTracker` class.

**Per-channel:**
- Independent Kalman filters (different channels may have different noise profiles)
- Independent adaptive peak detectors
- Each produces: rate_hz(t), confidence(t)

**Fusion strategies (test all 3, pick best):**

**Strategy A — Confidence-weighted mean:**
```
fused_rate = sum(w_i * rate_i) / sum(w_i)
where w_i = confidence_i / variance_i
```

**Strategy B — Agreement-gated weighted mean:**
```
Step 1: identify channels that agree (within 10% of median)
Step 2: confidence-weighted mean of agreeing channels only
Step 3: if < 2 channels agree, fall back to highest-confidence channel
```

**Strategy C — Hierarchical (spectral anchor + peak correction):**
```
Step 1: spectral consensus rate = confidence-weighted spectral across channels
Step 2: per-channel peak deviations from consensus
Step 3: if multiple channels show same deviation → accept (real transient)
         if only one channel deviates → reject (noise/artifact)
```
This is the most promising for transient detection: a real breath hold will
show up on ALL channels simultaneously.

**Channel weighting priors (from Phase 1 evidence):**
- Resp: diff ≈ CLE ≈ CRE ≈ avg ≈ CH (spectral MAE ~identical across channels)
- Card: avg > CLE > CH > CRE > diff (from CWT results)
- Weight by dynamic quality, not fixed channel preference

### Phase 3: Evaluation against GT

**Goal:** Quantify improvement over single-method baselines.

**Script:** `scripts/run_hybrid_rate_evaluation.py`

**Metrics (per session, per band):**
1. **Epoch-level MAE** (30s windows, compare to Phase 1 results)
2. **Sub-epoch tracking** (5s windows within 30s epochs — new)
3. **Transient detection** (compare innovations against known events):
   - Apnea events (from PSG annotations)
   - Stage transitions (from hypnogram)
   - Motion events (from accelerometer)
4. **Latency**: how quickly does the tracker respond to a rate change?
   - Simulate step change: epoch N at 15 br/min, epoch N+1 at 12 br/min
   - Measure: how many seconds until tracker is within 0.5 br/min of new rate?
5. **Computational cost**: wall-clock time per epoch (target: < 10ms for real-time)

**Comparison baselines:**
| Method | Description |
|--------|------------|
| spectral-only | Current best (Phase 1) |
| peaks-only | Previous best (peaks/k) |
| hilbert-only | Instantaneous frequency |
| simple-avg | (spectral + peaks) / 2 |
| **hybrid-single** | Kalman on best single channel |
| **hybrid-multi** | Multi-channel Kalman fusion |
| oracle | Best of spectral/peaks per epoch (floor: 1.06 br/min) |

**Plots (save to `writeup/figures/rate_hybrid/`):**
1. MAE comparison bar chart (all baselines + hybrid)
2. Example time series: 30 min excerpt showing tracker vs GT with transient events
3. Bland-Altman for hybrid method
4. Per-stage MAE comparison
5. Transient detection ROC (if we have apnea annotations)
6. Latency histogram

### Phase 4: Real-time demo pipeline

**Goal:** Streaming-mode pipeline that processes data sample-by-sample (or in small chunks).

**Module:** `sleep_monitor/realtime.py`

**Class: `RealtimeRateEstimator`**
```python
class RealtimeRateEstimator:
    def __init__(self, fs=100, channels=['CLE','CRE','CH']):
        # Initialize per-channel trackers
        # Initialize circular buffers for windowed spectral
        
    def update(self, sample_dict: dict) -> dict:
        """Process one new sample (or small chunk).
        
        Parameters
        ----------
        sample_dict : {'CLE': float, 'CRE': float, 'CH': float, 
                       'aX': float, 'aY': float, 'aZ': float}
        
        Returns
        -------
        dict with keys:
            resp_rate_hz, resp_confidence, resp_source
            card_rate_hz, card_confidence, card_source
            transient_flag (bool)
        """
        # 1. Append to circular buffer
        # 2. Online artifact removal (recursive OLS)
        # 3. Online bandpass (IIR filter, not FIR — causal)
        # 4. Peak detection on latest samples
        # 5. Every 30s: spectral update
        # 6. Kalman prediction + measurement updates
        # 7. Cross-channel fusion
        # Return current estimates
```

**Key real-time adaptations:**
- **IIR bandpass** (Butterworth, causal) instead of FIR (zero-phase is non-causal)
- **Recursive OLS** for artifact removal (update coefficients online)
- **Circular buffer** for spectral (keeps last 30s, overwrites oldest)
- **No Viterbi** (requires full sequence — Kalman gives similar smoothing causally)

**Demo script:** `scripts/demo_realtime.py`
- Reads a session file
- Feeds samples one-by-one to `RealtimeRateEstimator`
- Plots live-updating rate trace vs GT (matplotlib animation or terminal output)
- Shows: estimated rate, confidence band, transient flags, channel contributions

### Phase 5: Paper figures and reporting

**Goal:** Generate publication-ready figures comparing hybrid vs all baselines.

**Script:** `scripts/paper_hybrid_rate_figures.py`

**Figures for paper:**
1. **Architecture diagram** — block diagram of the hybrid pipeline (manually drawn or tikz)
2. **MAE improvement chart** — spectral → hybrid → hybrid+Viterbi progression
3. **Example tracings** — 2-3 sessions, showing GT vs hybrid with transient annotations
4. **Multi-channel contribution** — stacked area chart showing which channel dominates over time
5. **k-free vs k-scaled** — show that hybrid narrows the gap (less dependent on k)

**Expected outcomes:**
- Resp MAE: target 1.1-1.3 br/min (between spectral 1.54 and oracle 1.06)
- Card MAE: target improvement over spectral through CWT+Kalman fusion
- Real-time latency: < 5s for detecting a 2 br/min rate change
- Transient detection: identify apnea events and stage transitions

## Technical notes

### Why Kalman and not just weighted average?

Weighted average treats each window independently. Kalman:
1. Carries state between updates → temporal continuity without Viterbi
2. Models rate dynamics (rate + rate_derivative) → anticipates changes
3. Adaptive weighting: when spectral and peaks disagree, the one closer to the
   predicted state gets more weight (innovation-based gating)
4. Natural uncertainty quantification → confidence intervals for free
5. Handles asynchronous updates: spectral every 30s, peaks per-event

### Why not just Hilbert instantaneous frequency?

Hilbert gives instantaneous frequency but:
1. Very noisy — requires heavy post-smoothing (which removes transient info)
2. Edge effects at window boundaries
3. Mode mixing: if resp + cardiac both present, Hilbert tracks the dominant one
4. No natural confidence measure

The Kalman tracker uses Hilbert-like information (via adaptive peak detection)
but with principled noise handling and multi-source fusion.

### k-factor in the hybrid pipeline

Three options:
1. **No k**: evaluate hybrid at raw rates (compare to spectral no-k = 1.54)
2. **Global k per session**: same as Phase 6, applied after fusion
3. **Adaptive k**: track k(t) using a slow Kalman filter on the rate ratio
   (requires GT during calibration period, then runs autonomously)

For the paper: report all three. For the demo: use option 2 (brief calibration).

## File inventory

| File | Description |
|------|------------|
| `sleep_monitor/peaks_adaptive.py` | Adaptive peak detector with spectral prior |
| `sleep_monitor/rate_tracker.py` | Kalman filter rate tracker (single + multi-channel) |
| `sleep_monitor/realtime.py` | Streaming-mode real-time estimator |
| `scripts/run_hybrid_rate_evaluation.py` | Offline evaluation on all 12 sessions |
| `scripts/demo_realtime.py` | Real-time demo with live plotting |
| `scripts/paper_hybrid_rate_figures.py` | Paper-ready figures |
| `writeup/figures/rate_hybrid/` | Output figures |
| `reports/rates/hybrid_*.csv` | Detailed results |
| `artifacts/rate_hybrid_epochs.parquet` | Epoch-level hybrid estimates |

## Execution order

1. Phase 0 → Phase 1 → test on S1N1 only
2. Phase 2 → test multi-channel on S1N1
3. Phase 3 → full evaluation on all 12 sessions
4. Phase 4 → real-time demo
5. Phase 5 → paper figures

Phases 0-1 are the core; 2-5 build on them. If Phase 1 shows improvement on S1N1,
proceed. If not, revisit the peak detector (Phase 0) or Kalman tuning.

## Dependencies

- numpy, scipy (Kalman filter, peak detection, IIR filters)
- matplotlib (plotting, demo animation)
- sleep_monitor package (existing: filters, preprocessing, rates, quality, ground_truth)
- No new external dependencies needed

## Relation to existing pipeline

This replaces the "best single method" selection from Phase 1. The consolidation
pipeline (run_rate_consolidation.py) evaluated methods independently; this pipeline
fuses them intelligently. The Viterbi smoothing (Phase 4 of consolidation) becomes
optional because the Kalman filter provides similar temporal regularization causally.

The multi-channel fusion (Phase 2 of consolidation) is subsumed by the cross-channel
Kalman fusion here, which is more principled (per-channel dynamic quality vs static weights).
