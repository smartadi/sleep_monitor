# Validation Study — Results

## 1. Dataset Summary

12 sessions (6 subjects x 2 nights), totalling approximately _____ hours of recording and _____ 30-second epochs evaluated.

## 2. Calibration Factors

| Session | Subject | k_resp | k_cardiac |
|---------|---------|--------|-----------|
| S1N1    | OS001   |        |           |
| S1N2    | OS001   |        |           |
| S2N1    | OS002   |        |           |
| S2N2    | OS002   |        |           |
| S3N1    | OS003   |        |           |
| S3N2    | OS003   |        |           |
| S4N1    | OS004   |        |           |
| S4N2    | OS004   |        |           |
| S5N1    | OS005   |        |           |
| S5N2    | OS005   |        |           |
| S6N1    | OS006   |        |           |
| S6N2    | OS006   |        |           |
| **Median** | — |        |           |

*Fill from artifacts/validation_session.csv*

Notes on k stability:
- k_resp range: [ ___ , ___ ], subject clustering: ___
- k_cardiac range: [ ___ , ___ ], max night-to-night |delta k|: ___


## 3. Aggregate Accuracy

| Metric | Respiratory (br/min) | Cardiac (BPM) |
|--------|---------------------|---------------|
| MAE    |                     |               |
| RMSE   |                     |               |
| Bias   |                     |               |
| Pearson r |                  |               |
| p50    |                     |               |
| p90    |                     |               |
| Coverage |                   |               |
| N epochs |                   |               |

*Fill from the ALL row in artifacts/validation_session.csv*


## 4. Per-Session Accuracy

### 4.1 Summary Table

![Per-session metrics table](plots/rate_analysis/validation_summary_table.png)

*Commentary:*
- Best/worst session for respiratory: ___
- Best/worst session for cardiac: ___
- Any outlier sessions to discuss (e.g. S6N2 k_cardiac anomaly): ___

### 4.2 Per-Session MAE Bar Chart

![Per-session MAE](plots/rate_analysis/validation_session_mae.png)

*Commentary on session-to-session variability:*


## 5. Agreement Analysis

### 5.1 Bland-Altman Plots

![Bland-Altman](plots/rate_analysis/validation_bland_altman.png)

**Respiratory rate:**
- Bias: ___ br/min
- 95% Limits of agreement: [ ___ , ___ ] br/min
- Proportional bias observed? (does error scale with rate magnitude?)
- Stage-dependent patterns visible?

**Cardiac rate:**
- Bias: ___ BPM
- 95% Limits of agreement: [ ___ , ___ ] BPM
- Proportional bias observed?
- Stage-dependent patterns visible?

### 5.2 Scatter Plots

![Scatter agreement](plots/rate_analysis/validation_scatter.png)

*Commentary:*
- Spread around identity line
- Any systematic deviation at high/low rates
- Stage clustering visible?


## 6. Sleep Stage Stratified Accuracy

### 6.1 Per-Stage Metrics Table

| Stage | Resp MAE | Resp r | Resp n | Card MAE | Card r | Card n |
|-------|----------|--------|--------|----------|--------|--------|
| Wake  |          |        |        |          |        |        |
| N1    |          |        |        |          |        |        |
| N2    |          |        |        |          |        |        |
| N3    |          |        |        |          |        |        |
| REM   |          |        |        |          |        |        |

*Fill from artifacts/validation_stage.csv*

### 6.2 Error Distribution by Stage

![Stage boxplots](plots/rate_analysis/validation_stage_boxplots.png)

*Commentary:*
- Which stages have lowest/highest error?
- Hypothesis: Wake/N1 noisier due to movement and arousal; N2/N3 quieter
- REM: increased HRV and irregular breathing — does this affect accuracy?
- Connection to k biomarker findings (k_cardiac varies by stage: N1=1.71 > N2/N3=1.65 > Wake=1.61 > REM=1.58)


## 7. Discussion Points

### 7.1 Key Findings
- [ ] CAP-derived respiratory rate achieves MAE of ___ br/min (30s epochs, 12 sessions)
- [ ] CAP-derived cardiac rate achieves MAE of ___ BPM (30s epochs, 12 sessions)
- [ ] Both signals are reliably present in the capacitive sensor data
- [ ] Sleep stage modulates accuracy — describe pattern

### 7.2 Calibration Requirement
- Per-session k calibration is needed (k varies across subjects/nights)
- 50 random 30s windows (~25 min of data) sufficient for calibration
- Clinical deployment: could calibrate during attended setup period

### 7.3 Limitations
- Small cohort (6 subjects, healthy)
- k calibration requires paired PSG data — not available in standalone deployment
- OLS artifact removal assumes stationary accelerometer coupling
- Coverage < 100%: some epochs produce invalid estimates

### 7.4 Comparison to Prior Work
- Compare to published BCG-based sleep monitoring (bed sensors, wearables)
- Typical BCG cardiac MAE: 2–8 BPM depending on sensor and cohort
- Typical BCG respiratory MAE: 1–4 br/min
- Where does this system fall?


## 8. Next Steps After Validation

- [ ] Quality gating + median smoothing (Task 6 item 1) to reduce MAE from noisy epochs
- [ ] NLMS accelerometer removal to handle posture-dependent coupling
- [ ] Investigate whether default k (population median) is viable without per-session calibration
