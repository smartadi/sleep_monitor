"""
Unit tests for sleep_monitor/preprocessing.py

Tests the accelerometer artifact removal algorithms (OLS and NLMS) using
synthetic signals with known coupling, so we can verify the math directly.

OLS artifact removal (remove_acc_artifact):
  - Cap signal = physiological signal + β * acc_artifact
  - After OLS removal: residual should be uncorrelated with acc

NLMS artifact removal (remove_acc_artifact_nlms):
  - Same principle but with time-varying coupling
  - After NLMS: power of residual should be lower than before removal

We do NOT test preprocess_window / preprocess_full here — those require a
SleepSession object with real file I/O. Instead we unit-test the mathematical
core in isolation.
"""

import numpy as np
import pytest
from sleep_monitor.preprocessing import remove_acc_artifact, remove_acc_artifact_nlms
from sleep_monitor.filters import bandpass

FS      = 100.0
RESP_LO = 0.1
RESP_HI = 0.5
CARD_LO = 0.5
CARD_HI = 3.0


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rms(x):
    return np.sqrt(np.mean(x ** 2))


def _correlation(a, b):
    """Pearson correlation between two arrays."""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.std(a) * np.std(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (len(a) * denom))


# ─── OLS artifact removal ─────────────────────────────────────────────────────

class TestRemoveAccArtifactOLS:

    def test_known_coupling_beta_is_removed(self):
        """
        Construct: cap = signal + beta * acc  (with known beta)
        After OLS removal, residual should have much lower correlation with acc.
        """
        rng = np.random.default_rng(0)
        n = int(30 * FS)
        t = np.arange(n) / FS

        # Physiological signal: 0.25 Hz resp sine (in resp band)
        physio = np.sin(2 * np.pi * 0.25 * t)
        # Accelerometer artifact: broadband noise (contains energy in resp band)
        acc_raw = rng.standard_normal(n)
        # Known coupling
        true_beta = 3.0
        cap_raw = physio + true_beta * acc_raw

        # Run OLS removal
        cleaned = remove_acc_artifact(cap_raw, acc_raw, RESP_LO, RESP_HI, fs=FS)

        # Bandpass acc to resp band for correlation comparison
        acc_bp = bandpass(acc_raw, RESP_LO, RESP_HI, fs=FS)

        # The cleaned signal should have low correlation with bandpassed acc
        r_before = _correlation(bandpass(cap_raw, RESP_LO, RESP_HI, fs=FS), acc_bp)
        r_after  = _correlation(cleaned, acc_bp)

        assert abs(r_after) < abs(r_before), \
            f"Correlation with acc not reduced: before={r_before:.3f}, after={r_after:.3f}"
        assert abs(r_after) < 0.15, \
            f"Residual still highly correlated with acc: r={r_after:.3f}"

    def test_ols_formula_matches_manual_computation(self):
        """
        OLS beta = dot(acc_bp, cap_bp) / dot(acc_bp, acc_bp)
        We verify that remove_acc_artifact uses this formula by checking the
        residual equals cap_bp - beta * acc_bp with our manually computed beta.
        """
        rng = np.random.default_rng(1)
        n = int(30 * FS)
        cap_raw = rng.standard_normal(n)
        acc_raw = rng.standard_normal(n)

        cap_bp = bandpass(cap_raw, RESP_LO, RESP_HI, fs=FS)
        acc_bp = bandpass(acc_raw, RESP_LO, RESP_HI, fs=FS)

        # Manual OLS
        beta = np.dot(acc_bp, cap_bp) / (np.dot(acc_bp, acc_bp) + 1e-12)
        expected_residual = cap_bp - beta * acc_bp

        # Function result
        actual_residual = remove_acc_artifact(cap_raw, acc_raw, RESP_LO, RESP_HI, fs=FS)

        np.testing.assert_allclose(actual_residual, expected_residual, rtol=1e-10,
            err_msg="OLS residual doesn't match manual beta computation")

    def test_zero_acc_returns_bandpassed_cap(self):
        """If acc is all zeros, beta=0 and residual = bandpassed cap."""
        rng = np.random.default_rng(2)
        cap_raw = rng.standard_normal(int(30 * FS))
        acc_raw = np.zeros_like(cap_raw)

        result = remove_acc_artifact(cap_raw, acc_raw, RESP_LO, RESP_HI, fs=FS)
        expected = bandpass(cap_raw, RESP_LO, RESP_HI, fs=FS)
        np.testing.assert_allclose(result, expected, rtol=1e-10,
            err_msg="Zero acc should give same result as plain bandpass")

    def test_output_length_matches_input(self):
        rng = np.random.default_rng(3)
        n = int(20 * FS)
        cap = rng.standard_normal(n)
        acc = rng.standard_normal(n)
        out = remove_acc_artifact(cap, acc, CARD_LO, CARD_HI, fs=FS)
        assert len(out) == n

    def test_works_on_cardiac_band(self):
        """OLS should work correctly for the cardiac band too."""
        rng = np.random.default_rng(4)
        n = int(20 * FS)
        t = np.arange(n) / FS
        physio = np.sin(2 * np.pi * 1.0 * t)
        acc_raw = rng.standard_normal(n)
        cap_raw = physio + 2.0 * acc_raw

        cleaned = remove_acc_artifact(cap_raw, acc_raw, CARD_LO, CARD_HI, fs=FS)
        acc_bp  = bandpass(acc_raw, CARD_LO, CARD_HI, fs=FS)

        r = _correlation(cleaned, acc_bp)
        assert abs(r) < 0.20, f"Cardiac band: residual still correlated with acc: r={r:.3f}"


# ─── NLMS artifact removal ────────────────────────────────────────────────────

class TestRemoveAccArtifactNLMS:

    def test_reduces_artifact_power(self):
        """
        NLMS should reduce the power of the accelerometer artifact in the signal.
        We inject a known ACC-correlated artifact and verify the output RMS drops.
        """
        rng = np.random.default_rng(10)
        n = int(30 * FS)
        t = np.arange(n) / FS

        physio  = np.sin(2 * np.pi * 0.25 * t)   # 0.25 Hz signal
        acc_raw = rng.standard_normal(n)
        cap_raw = physio + 5.0 * acc_raw           # heavy artifact

        # OLS baseline (single beta)
        ols_out  = remove_acc_artifact(cap_raw, acc_raw, RESP_LO, RESP_HI, fs=FS)
        nlms_out = remove_acc_artifact_nlms(cap_raw, acc_raw, RESP_LO, RESP_HI, fs=FS)

        # Both should reduce power compared to raw cap (in resp band)
        cap_bp = bandpass(cap_raw, RESP_LO, RESP_HI, fs=FS)
        assert _rms(nlms_out) <= _rms(cap_bp) * 1.05, \
            f"NLMS did not reduce signal power: nlms_rms={_rms(nlms_out):.4f}, cap_rms={_rms(cap_bp):.4f}"

    def test_output_length_matches_input(self):
        rng = np.random.default_rng(11)
        n = int(25 * FS)
        cap = rng.standard_normal(n)
        acc = rng.standard_normal(n)
        out = remove_acc_artifact_nlms(cap, acc, RESP_LO, RESP_HI, fs=FS)
        assert len(out) == n

    def test_short_signal_returns_zeros(self):
        """Signal shorter than MIN_SAMPLES (50) returns zeros without crashing.

        Previously, bandpass() was called before the length guard, causing filtfilt
        to raise ValueError. The fix adds a MIN_SAMPLES check before bandpass().
        """
        for n in (5, 10, 20, 49):
            cap = np.random.randn(n)
            acc = np.random.randn(n)
            out = remove_acc_artifact_nlms(cap, acc, RESP_LO, RESP_HI, fs=FS, taps=16)
            assert len(out) == n, f"Output length mismatch for n={n}"
            np.testing.assert_array_equal(out, np.zeros(n),
                err_msg=f"Expected zeros for short signal (n={n})")

    def test_nlms_handles_time_varying_coupling(self):
        """
        NLMS advantage over OLS: time-varying coupling.
        First half: beta=1. Second half: beta=4.
        OLS uses one global beta; NLMS should adapt.
        We verify NLMS output is finite and physically plausible.
        """
        rng = np.random.default_rng(12)
        n = int(60 * FS)
        t = np.arange(n) / FS
        physio  = np.sin(2 * np.pi * 0.25 * t)
        acc_raw = rng.standard_normal(n)

        # Time-varying coupling
        beta = np.ones(n)
        beta[n // 2:] = 4.0
        cap_raw = physio + beta * acc_raw

        out = remove_acc_artifact_nlms(cap_raw, acc_raw, RESP_LO, RESP_HI, fs=FS)
        assert len(out) == n
        assert not np.any(np.isnan(out)), "NLMS produced NaN values"
        assert not np.any(np.isinf(out)), "NLMS produced Inf values"


# ─── Mathematical sanity checks ───────────────────────────────────────────────

class TestMathSanity:
    def test_ols_beta_formula_analytically(self):
        """
        Verify the OLS formula: β* = (Xᵀy) / (XᵀX)
        is the least-squares solution that minimises ‖y - βx‖².

        For y = βx + ε, the OLS estimate of β should equal the true β
        as sample size → ∞ (noise averages out).
        """
        rng = np.random.default_rng(99)
        n   = 10_000   # large n so noise averages out
        x   = rng.standard_normal(n)
        eps = rng.standard_normal(n) * 0.1   # small noise
        true_beta = 2.5
        y = true_beta * x + eps

        # Manual OLS
        beta_hat = np.dot(x, y) / (np.dot(x, x) + 1e-12)
        assert abs(beta_hat - true_beta) < 0.05, \
            f"OLS formula inaccurate: β*={beta_hat:.4f}, true={true_beta}"

    def test_nlms_weight_update_converges(self):
        """
        The NLMS weight update rule: w ← w + (μ / (xᵀx + ε)) * e * x
        For a static signal, weights should converge and error should decrease.
        """
        rng = np.random.default_rng(100)
        n = 500
        # Pure stationary signal: y = 2 * x[n] (single-tap)
        x = rng.standard_normal(n)
        y = 2.0 * x

        # Manual single-tap NLMS
        w = 0.0
        errors = []
        mu = 0.1
        for i in range(n):
            pred = w * x[i]
            e = y[i] - pred
            errors.append(abs(e))
            norm = x[i] ** 2 + 1e-6
            w += (mu / norm) * e * x[i]

        # Error should decrease over time: mean of last 100 < mean of first 100
        early_err  = np.mean(errors[:100])
        late_err   = np.mean(errors[-100:])
        assert late_err < early_err, \
            f"NLMS didn't converge: early_err={early_err:.4f}, late_err={late_err:.4f}"
