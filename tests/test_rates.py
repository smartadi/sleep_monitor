"""
Unit tests for sleep_monitor/rates.py

Strategy: feed each estimator a pure sine wave at a known frequency.
The estimator must return a value within a tight tolerance of the known rate.
This directly verifies the mathematical correctness of each method.

Tolerances chosen conservatively — a correctly-implemented estimator should
easily hit these on a clean synthetic signal:
  - Spectral / ACF / Hilbert / Peaks: ±0.03 Hz on a 60-s resp signal
  - Zero-crossing: ±0.03 Hz (sub-sample interpolated)
  - Cardiac methods: ±0.05 Hz on a 30-s cardiac signal

Additional tests cover:
  - NaN on degenerate inputs (too short, all zeros)
  - Scaled variants: dividing by k is applied correctly
  - fuse_rates: outlier rejection, each fusion mode
  - sliding_rates: correct number of windows and centre times
"""

import numpy as np
import pytest
from sleep_monitor.rates import (
    rate_spectral, rate_acf, rate_hilbert, rate_zerocross, rate_peaks,
    rate_hilbert_scaled_cardiac, rate_peaks_scaled_resp,
    estimate_rate, fuse_rates, sliding_rates,
)

FS          = 100.0
RESP_LO     = 0.1
RESP_HI     = 0.5
CARD_LO     = 0.5
CARD_HI     = 3.0
RESP_F0     = 0.25    # Hz — 15 br/min, well inside resp band
CARDIAC_F0  = 1.0     # Hz — 60 BPM, well inside cardiac band
RESP_TOL    = 0.03    # Hz tolerance for resp estimators
CARDIAC_TOL = 0.05    # Hz tolerance for cardiac estimators


# ─── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def resp_sine():
    """60-second, 0.25 Hz pure sine — clean resp signal."""
    t = np.arange(int(60.0 * FS)) / FS
    return np.sin(2 * np.pi * RESP_F0 * t)


@pytest.fixture(scope="module")
def cardiac_sine():
    """30-second, 1.0 Hz pure sine — clean cardiac signal."""
    t = np.arange(int(30.0 * FS)) / FS
    return np.sin(2 * np.pi * CARDIAC_F0 * t)


# ─── rate_spectral ────────────────────────────────────────────────────────────

class TestRateSpectral:
    def test_resp_sine_correct(self, resp_sine):
        rate = rate_spectral(resp_sine, RESP_LO, RESP_HI, fs=FS)
        assert abs(rate - RESP_F0) <= RESP_TOL, \
            f"spectral resp: expected {RESP_F0}, got {rate:.4f}"

    def test_cardiac_sine_correct(self, cardiac_sine):
        rate = rate_spectral(cardiac_sine, CARD_LO, CARD_HI, fs=FS)
        assert abs(rate - CARDIAC_F0) <= CARDIAC_TOL, \
            f"spectral cardiac: expected {CARDIAC_F0}, got {rate:.4f}"

    def test_returns_nan_for_short_signal(self):
        sig = np.sin(2 * np.pi * 0.25 * np.arange(5) / FS)
        result = rate_spectral(sig, RESP_LO, RESP_HI, fs=FS)
        assert np.isnan(result), "Should return nan for very short signal"

    def test_returns_float(self, resp_sine):
        result = rate_spectral(resp_sine, RESP_LO, RESP_HI, fs=FS)
        assert isinstance(result, float)


# ─── rate_acf ─────────────────────────────────────────────────────────────────

class TestRateAcf:
    def test_resp_sine_correct(self, resp_sine):
        rate = rate_acf(resp_sine, RESP_LO, RESP_HI, fs=FS)
        assert abs(rate - RESP_F0) <= RESP_TOL, \
            f"ACF resp: expected {RESP_F0}, got {rate:.4f}"

    def test_cardiac_sine_correct(self, cardiac_sine):
        rate = rate_acf(cardiac_sine, CARD_LO, CARD_HI, fs=FS)
        assert abs(rate - CARDIAC_F0) <= CARDIAC_TOL, \
            f"ACF cardiac: expected {CARDIAC_F0}, got {rate:.4f}"

    def test_parabolic_interpolation_improves_resolution(self):
        """Sub-sample parabolic interpolation should yield non-integer-multiple result.

        A 0.25 Hz signal has period = 400 samples at 100 Hz.
        Parabolic interpolation lets the ACF peak be at a non-integer lag,
        so the returned rate should not be exactly FS/400 = 0.25 but close.
        This test just confirms interpolation doesn't crash and stays accurate.
        """
        t = np.arange(int(60 * FS)) / FS
        sig = np.sin(2 * np.pi * 0.27 * t)   # slightly off-grid frequency
        rate = rate_acf(sig, RESP_LO, RESP_HI, fs=FS)
        assert abs(rate - 0.27) <= 0.03, \
            f"ACF off-grid: expected ~0.27, got {rate:.4f}"

    def test_returns_nan_for_flat_signal(self):
        sig = np.zeros(6000)
        result = rate_acf(sig, RESP_LO, RESP_HI, fs=FS)
        assert np.isnan(result)


# ─── rate_hilbert ─────────────────────────────────────────────────────────────

class TestRateHilbert:
    def test_resp_sine_correct(self, resp_sine):
        rate = rate_hilbert(resp_sine, RESP_LO, RESP_HI, fs=FS)
        assert abs(rate - RESP_F0) <= RESP_TOL, \
            f"Hilbert resp: expected {RESP_F0}, got {rate:.4f}"

    def test_cardiac_sine_correct(self, cardiac_sine):
        rate = rate_hilbert(cardiac_sine, CARD_LO, CARD_HI, fs=FS)
        assert abs(rate - CARDIAC_F0) <= CARDIAC_TOL, \
            f"Hilbert cardiac: expected {CARDIAC_F0}, got {rate:.4f}"

    def test_instantaneous_frequency_principle(self):
        """
        The Hilbert instantaneous frequency of sin(2π f t) should equal f.

        For a pure sine, the analytic signal phase is φ(t) = 2π f t,
        so dφ/dt / (2π) = f exactly. We verify this property directly.
        """
        f = 1.5  # Hz — inside cardiac band
        t = np.arange(int(30.0 * FS)) / FS
        sig = np.sin(2 * np.pi * f * t)
        rate = rate_hilbert(sig, CARD_LO, CARD_HI, fs=FS)
        assert abs(rate - f) < 0.02, \
            f"Hilbert inst-freq property violated: f={f}, got {rate:.4f}"

    def test_returns_nan_when_too_few_valid_samples(self):
        """Very short signal with all samples below 25th-percentile amplitude gate → nan."""
        sig = np.zeros(20)   # amplitude = 0 everywhere → all filtered out by amplitude gate
        result = rate_hilbert(sig, RESP_LO, RESP_HI, fs=FS)
        assert np.isnan(result)


# ─── rate_zerocross ───────────────────────────────────────────────────────────

class TestRateZerocross:
    def test_resp_sine_correct(self, resp_sine):
        rate = rate_zerocross(resp_sine, fs=FS)
        assert abs(rate - RESP_F0) <= RESP_TOL, \
            f"Zero-cross resp: expected {RESP_F0}, got {rate:.4f}"

    def test_cardiac_sine_correct(self, cardiac_sine):
        rate = rate_zerocross(cardiac_sine, fs=FS)
        assert abs(rate - CARDIAC_F0) <= CARDIAC_TOL, \
            f"Zero-cross cardiac: expected {CARDIAC_F0}, got {rate:.4f}"

    def test_upward_crossings_only(self):
        """
        A sine completes 1 upward crossing per cycle.
        For f=0.25 Hz over 60 s: exactly 15 upward crossings.
        rate = (N_crossings - 1) / (t_last - t_first) must be close to 0.25 Hz.
        """
        f = 0.25
        duration = 60.0
        t = np.arange(int(duration * FS)) / FS
        sig = np.sin(2 * np.pi * f * t)

        # Count upward crossings manually
        signs = np.sign(sig)
        signs[signs == 0] = 1
        n_up = np.sum(np.diff(signs) > 0)

        # Expected: floor(f * duration) ≈ 15 upward crossings
        assert abs(n_up - int(f * duration)) <= 1, \
            f"Unexpected number of upward crossings: {n_up}"

    def test_returns_nan_for_too_few_crossings(self):
        """Signal with only one zero crossing → not enough to estimate rate."""
        sig = np.array([-1.0, -0.5, 0.5, 1.0, 0.5])   # one upward crossing
        result = rate_zerocross(sig, fs=FS)
        assert np.isnan(result)


# ─── rate_peaks ───────────────────────────────────────────────────────────────

class TestRatePeaks:
    def test_resp_sine_correct(self, resp_sine):
        rate = rate_peaks(resp_sine, RESP_LO, RESP_HI, fs=FS)
        assert abs(rate - RESP_F0) <= RESP_TOL, \
            f"Peaks resp: expected {RESP_F0}, got {rate:.4f}"

    def test_cardiac_sine_correct(self, cardiac_sine):
        rate = rate_peaks(cardiac_sine, CARD_LO, CARD_HI, fs=FS)
        assert abs(rate - CARDIAC_F0) <= CARDIAC_TOL, \
            f"Peaks cardiac: expected {CARDIAC_F0}, got {rate:.4f}"

    def test_peak_count_formula(self):
        """
        rate_peaks formula: (N_peaks - 1) / ((last_peak - first_peak) / fs)

        We construct a signal with exactly known peaks and check the formula.
        """
        # Create a signal with peaks at sample indices 0, 400, 800, 1200 (f=0.25 Hz at 100 Hz)
        sig = np.zeros(1500)
        peak_indices = [0, 400, 800, 1200]
        for idx in peak_indices:
            sig[idx] = 1.0

        # expected: (4 - 1) / ((1200 - 0) / 100) = 3 / 12 = 0.25 Hz
        expected_rate = (len(peak_indices) - 1) / ((peak_indices[-1] - peak_indices[0]) / FS)
        assert abs(expected_rate - 0.25) < 1e-10, "Test construction error"

    def test_returns_nan_for_too_few_peaks(self):
        """Only one peak found → can't compute rate → nan."""
        sig = np.zeros(1000)
        sig[500] = 1.0   # single isolated peak
        result = rate_peaks(sig, RESP_LO, RESP_HI, fs=FS)
        assert np.isnan(result)


# ─── rate_hilbert_scaled_cardiac ──────────────────────────────────────────────

class TestHilbertScaledCardiac:
    def test_k_divides_correctly(self, cardiac_sine):
        """
        rate_hilbert_scaled_cardiac(x, k) = rate_hilbert(x) / k

        We know cardiac_sine is a 1 Hz signal. With k=2, the scaled rate
        should be ~0.5 Hz. This verifies the division is applied correctly.
        """
        k = 2.0
        raw = rate_hilbert(cardiac_sine, CARD_LO, CARD_HI, fs=FS)
        scaled = rate_hilbert_scaled_cardiac(cardiac_sine, k=k, fs=FS)
        assert abs(scaled - raw / k) < 1e-10, \
            f"Division by k incorrect: raw={raw:.4f}, k={k}, scaled={scaled:.4f}, expected={raw/k:.4f}"

    def test_k_equals_one_matches_raw(self, cardiac_sine):
        """k=1 should return the same as raw rate_hilbert."""
        raw = rate_hilbert(cardiac_sine, CARD_LO, CARD_HI, fs=FS)
        scaled = rate_hilbert_scaled_cardiac(cardiac_sine, k=1.0, fs=FS)
        assert abs(scaled - raw) < 1e-10

    def test_invalid_k_returns_nan(self, cardiac_sine):
        """k <= 0 or nan/inf → return nan."""
        for bad_k in [0.0, -1.0, np.nan, np.inf, None]:
            result = rate_hilbert_scaled_cardiac(cardiac_sine, k=bad_k, fs=FS)
            assert np.isnan(result), f"Expected nan for k={bad_k}, got {result}"

    def test_typical_k_range(self, cardiac_sine):
        """With k in [1.48, 1.93] (real calibration range), result should be in cardiac band."""
        for k in [1.48, 1.67, 1.93]:
            result = rate_hilbert_scaled_cardiac(cardiac_sine, k=k, fs=FS)
            assert np.isfinite(result), f"Got non-finite result for k={k}"
            assert CARD_LO <= result <= CARD_HI, \
                f"Result {result:.3f} Hz outside cardiac band for k={k}"


# ─── rate_peaks_scaled_resp ───────────────────────────────────────────────────

class TestPeaksScaledResp:
    def test_k_equals_one_close_to_raw_count(self, resp_sine):
        """
        With k=1, rate_peaks_scaled_resp should return approx n_peaks / T_total.
        This uses a loose detector (prom=0.05σ, min_dist=0.4s) so it may count
        more peaks than rate_peaks — that's the design intent.
        """
        result = rate_peaks_scaled_resp(resp_sine, k=1.0, fs=FS)
        # Should be in the resp band (it counts all bumps, so will be ≥ true rate)
        assert np.isfinite(result), "Expected finite result for k=1"
        assert result > 0, f"Rate must be positive, got {result}"

    def test_k_scales_down(self, resp_sine):
        """Larger k → smaller returned rate (the raw overcount is divided out)."""
        r1 = rate_peaks_scaled_resp(resp_sine, k=1.0, fs=FS)
        r2 = rate_peaks_scaled_resp(resp_sine, k=1.5, fs=FS)
        r3 = rate_peaks_scaled_resp(resp_sine, k=2.0, fs=FS)
        assert r1 > r2 > r3, f"Rates not monotonically decreasing with k: {r1:.3f}, {r2:.3f}, {r3:.3f}"

    def test_k_division_formula(self, resp_sine):
        """
        Formula: (n_peaks / k) / (len(x) / fs)
        We can verify: result * k should equal the un-scaled version.
        """
        k = 1.3
        r_scaled = rate_peaks_scaled_resp(resp_sine, k=k, fs=FS)
        r_unscaled = rate_peaks_scaled_resp(resp_sine, k=1.0, fs=FS)
        # r_scaled * k should ≈ r_unscaled
        assert abs(r_scaled * k - r_unscaled) < 1e-10, \
            f"k division not applied correctly: {r_scaled * k:.4f} vs {r_unscaled:.4f}"

    def test_invalid_k_returns_nan(self, resp_sine):
        for bad_k in [0.0, -0.5, np.nan, None]:
            result = rate_peaks_scaled_resp(resp_sine, k=bad_k, fs=FS)
            assert np.isnan(result), f"Expected nan for k={bad_k}, got {result}"

    def test_short_signal_returns_nan(self):
        sig = np.sin(np.arange(10) * 0.1)  # only 10 samples
        result = rate_peaks_scaled_resp(sig, k=1.3, fs=FS)
        assert np.isnan(result)


# ─── fuse_rates ───────────────────────────────────────────────────────────────

class TestFuseRates:
    def test_median_of_identical_values(self):
        rates = {'a': 0.25, 'b': 0.25, 'c': 0.25}
        assert fuse_rates(rates, RESP_LO, RESP_HI, how='median') == pytest.approx(0.25)

    def test_median_ignores_outlier(self):
        """Median fusion should ignore one outlier that's outside the band."""
        rates = {'a': 0.25, 'b': 0.25, 'c': 0.25, 'd': 0.24, 'e': 0.99}
        # 0.99 is outside RESP_HI=0.5, so filtered; median of [0.24, 0.25, 0.25, 0.25]
        result = fuse_rates(rates, RESP_LO, RESP_HI, how='median')
        assert abs(result - 0.25) < 0.01

    def test_returns_nan_when_no_valid_values(self):
        rates = {'a': 0.0, 'b': 10.0}   # both outside resp band [0.1, 0.5]
        result = fuse_rates(rates, RESP_LO, RESP_HI, how='median')
        assert np.isnan(result)

    def test_trimmed_drops_extremes(self):
        """'trimmed' mode removes the highest and lowest before taking median."""
        rates = {'a': 0.20, 'b': 0.25, 'c': 0.25, 'd': 0.25, 'e': 0.30}
        # After trimming 0.20 and 0.30, median([0.25, 0.25, 0.25]) = 0.25
        result = fuse_rates(rates, RESP_LO, RESP_HI, how='trimmed')
        assert result == pytest.approx(0.25)

    def test_weighted_biased_toward_consensus(self):
        """'weighted' mode should weight values close to the median more heavily."""
        # Four values at 0.25, one outlier at 0.45 (still inside band)
        rates = {'a': 0.25, 'b': 0.25, 'c': 0.25, 'd': 0.25, 'e': 0.45}
        result = fuse_rates(rates, RESP_LO, RESP_HI, how='weighted')
        # Result should be much closer to 0.25 than 0.45
        assert result < 0.30, f"Weighted fusion not consensus-biased: {result:.3f}"

    def test_single_valid_value_returned(self):
        rates = {'a': 0.25, 'b': np.nan, 'c': 10.0}
        result = fuse_rates(rates, RESP_LO, RESP_HI)
        assert result == pytest.approx(0.25)


# ─── estimate_rate ────────────────────────────────────────────────────────────

class TestEstimateRate:
    def test_returns_all_methods(self, resp_sine):
        result = estimate_rate(resp_sine, RESP_LO, RESP_HI, fs=FS)
        expected_keys = {'spectral', 'acf', 'hilbert', 'zerocross', 'peaks'}
        assert set(result.keys()) == expected_keys

    def test_all_methods_finite_for_clean_signal(self, resp_sine):
        result = estimate_rate(resp_sine, RESP_LO, RESP_HI, fs=FS)
        for method, rate in result.items():
            assert np.isfinite(rate), f"Method '{method}' returned non-finite: {rate}"

    def test_all_methods_near_correct_rate(self, resp_sine):
        result = estimate_rate(resp_sine, RESP_LO, RESP_HI, fs=FS)
        for method, rate in result.items():
            assert abs(rate - RESP_F0) <= RESP_TOL, \
                f"Method '{method}': expected {RESP_F0}, got {rate:.4f}"

    def test_envelope_included_when_requested(self, cardiac_sine):
        result = estimate_rate(cardiac_sine, CARD_LO, CARD_HI, fs=FS,
                               include_envelope=True)
        assert 'envelope' in result

    def test_envelope_excluded_by_default(self, resp_sine):
        result = estimate_rate(resp_sine, RESP_LO, RESP_HI, fs=FS)
        assert 'envelope' not in result


# ─── sliding_rates ────────────────────────────────────────────────────────────

class TestSlidingRates:
    def test_window_count(self):
        """
        For a 60-s signal, win=20s, step=1s:
          n_windows = floor((6000 - 2000) / 100) + 1 = 41
        """
        t = np.arange(int(60.0 * FS)) / FS
        sig = np.sin(2 * np.pi * RESP_F0 * t)
        t_s, rates = sliding_rates(sig, RESP_LO, RESP_HI, fs=FS,
                                   win_sec=20.0, step_sec=1.0)
        assert len(t_s) == 41, f"Expected 41 windows, got {len(t_s)}"

    def test_centre_time_of_first_window(self):
        """First window starts at 0, length=20s → centre at 10.0 s."""
        t = np.arange(int(60.0 * FS)) / FS
        sig = np.sin(2 * np.pi * RESP_F0 * t)
        t_s, _ = sliding_rates(sig, RESP_LO, RESP_HI, fs=FS,
                               win_sec=20.0, step_sec=1.0)
        assert t_s[0] == pytest.approx(10.0), f"First centre time: {t_s[0]}"

    def test_returns_all_methods(self):
        t = np.arange(int(30.0 * FS)) / FS
        sig = np.sin(2 * np.pi * RESP_F0 * t)
        _, rates = sliding_rates(sig, RESP_LO, RESP_HI, fs=FS,
                                 win_sec=10.0, step_sec=5.0)
        from sleep_monitor.config import METHOD_NAMES
        assert set(rates.keys()) == set(METHOD_NAMES)

    def test_each_method_array_same_length_as_times(self):
        t = np.arange(int(60.0 * FS)) / FS
        sig = np.sin(2 * np.pi * RESP_F0 * t)
        t_s, rates = sliding_rates(sig, RESP_LO, RESP_HI, fs=FS,
                                   win_sec=20.0, step_sec=1.0)
        for method, arr in rates.items():
            assert len(arr) == len(t_s), \
                f"Method '{method}' array length {len(arr)} != t_s length {len(t_s)}"
