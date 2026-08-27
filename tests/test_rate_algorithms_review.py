"""
Reviewer test battery for the rate-estimation algorithms (2026-08-26).

These are not "does it work on a clean sine" tests — `test_rates.py` already
covers that. These encode the *paper-critical* properties and the failure modes
found in code review, so they will fail loudly if any of them is silently
"fixed" in a way that changes a published number, or if a latent bug is ever
exercised on real data.

Each test names the finding it guards (see
`writeup/edits/REVIEWER_METHODS_CODE_REVIEW_2026-08-26.md`).

Run: pytest tests/test_rate_algorithms_review.py -v
"""

import numpy as np
import pytest

from sleep_monitor.rates import (
    rate_spectral, rate_spectral_interp, rate_acf,
    rate_peaks, rate_peaks_scaled_resp,
)
from sleep_monitor.ground_truth import _quality_filter
from sleep_monitor.rate_metrics import (
    frequency_sweep, degeneracy_report, octave_report,
)

FS = 100.0
RESP_LO, RESP_HI = 0.1, 0.5
CARD_LO, CARD_HI = 0.5, 3.0


def _sine(f, dur_s, fs=FS):
    t = np.arange(int(dur_s * fs)) / fs
    return np.sin(2 * np.pi * f * t)


# ── C1: rate_spectral is degenerate in the respiratory band ───────────────────

class TestSpectralDegeneracy:
    """The estimator that produced the published resp 0.91 br/min is a constant.

    `test_rates.py` tests it only at 0.25 Hz — exactly the degenerate bin — so
    the existing suite passes *because* of the bug, not despite it. These tests
    pin the degeneracy so nobody quietly reintroduces the spectral estimator into
    a headline number believing it measures something.
    """

    def test_only_two_distinct_outputs_across_resp_band(self):
        sweep = frequency_sweep(rate_spectral, RESP_LO, RESP_HI, FS,
                                freqs=np.linspace(0.11, 0.48, 40))
        rep = degeneracy_report(sweep)
        assert rep['is_degenerate'], (
            "rate_spectral is documented as degenerate in the resp band; if this "
            "test fails the estimator changed — recheck every k and MAE it feeds.")
        assert rep['n_distinct_outputs'] <= 3, rep

    def test_returns_15_brmin_for_most_of_the_band(self):
        # 0.25 Hz == 15 br/min is the modal output for nearly the whole band
        rep = degeneracy_report(
            frequency_sweep(rate_spectral, RESP_LO, RESP_HI, FS,
                            freqs=np.linspace(0.11, 0.37, 30)))
        assert abs(rep['modal_output_hz'] - 0.25) < 1e-6
        assert rep['modal_fraction'] > 0.9, (
            f"expected ~all epochs at 0.25 Hz, got {rep['modal_fraction']:.2f}")

    def test_a_true_12_brmin_signal_is_reported_as_15(self):
        # A subject breathing 12 br/min (0.20 Hz) is reported at 15 br/min.
        est = rate_spectral(_sine(0.20, 60), RESP_LO, RESP_HI, FS)
        assert abs(est - 0.25) < 1e-6
        assert abs(est - 0.20) > 0.04

    def test_interp_variant_actually_tracks_the_band(self):
        # The documented replacement must NOT be degenerate.
        rep = degeneracy_report(
            frequency_sweep(rate_spectral_interp, RESP_LO, RESP_HI, FS,
                            freqs=np.linspace(0.11, 0.48, 40)))
        assert not rep['is_degenerate']
        assert rep['distinct_fraction'] > 0.8


# ── C2: rate_acf locks onto a subharmonic (reports half the rate) ─────────────

class TestAcfOctaveError:
    """`rate_acf` picks the maximum-*prominence* ACF peak, not the shortest
    qualifying period, so on part of the band it returns half the true rate.

    This is silent: half of a plausible breathing rate is still a plausible
    breathing rate. Guarding it documents the operating range where the ACF
    estimator is trustworthy and flags if it ever widens/narrows.
    """

    @pytest.mark.parametrize("f", [0.30, 0.34, 0.40])
    def test_half_rate_in_upper_resp_band(self, f):
        est = rate_acf(_sine(f, 60), RESP_LO, RESP_HI, FS)
        assert abs(est - f / 2.0) < 0.02, (
            f"expected the known subharmonic lock at {f} Hz; got {est:.3f}. "
            "If this now tracks correctly the estimator was fixed — good, but "
            "update the manuscript's resp estimator characterization.")

    @pytest.mark.parametrize("f", [0.15, 0.20, 0.25])
    def test_correct_in_lower_resp_band(self, f):
        est = rate_acf(_sine(f, 60), RESP_LO, RESP_HI, FS)
        assert abs(est - f) < 0.02

    def test_octave_report_flags_the_band(self):
        rep = octave_report(
            frequency_sweep(rate_acf, RESP_LO, RESP_HI, FS,
                            freqs=np.linspace(0.12, 0.48, 40)))
        assert rep['half_fraction'] > 0.2, (
            "a meaningful fraction of the resp band should show the subharmonic "
            f"lock; got {rep['half_fraction']:.2f}")

    def test_cardiac_band_upper_half_also_locks(self):
        # 2.0 Hz (120 BPM) reported as 1.0 Hz (60 BPM) — a clinically serious miss
        est = rate_acf(_sine(2.0, 30), CARD_LO, CARD_HI, FS)
        assert abs(est - 1.0) < 0.05


# ── C3: rate_peaks_scaled_resp uses a different rate definition than rate_peaks ─

class TestScaledRespDefinitionMismatch:
    """`rate_peaks` uses (N-1)/peak_span; `rate_peaks_scaled_resp` uses
    N/full_window. On a clean sine with k=1 they disagree, and the scaled one
    overcounts the true rate by a few percent from *definition alone* — an
    overcount that is then baked into the fitted respiratory k (~1.18).
    """

    def test_scaled_k1_does_not_equal_rate_peaks(self):
        x = _sine(0.25, 30)
        a = rate_peaks(x, RESP_LO, RESP_HI, FS)
        b = rate_peaks_scaled_resp(x, k=1.0, fs=FS)
        assert abs(a - b) > 0.01, (
            "if these now match the definition was unified — remove this guard "
            "and re-derive respiratory k, which currently absorbs the gap.")

    def test_scaled_k1_overcounts_the_true_rate(self):
        # true 0.25 Hz, 30 s window -> scaled reads high purely from N/T vs (N-1)/span
        b = rate_peaks_scaled_resp(_sine(0.25, 30), k=1.0, fs=FS)
        assert b > 0.25 + 0.005
        # the definitional inflation is on the order of 1 / (n_breaths) ~ 5-8%
        assert (b - 0.25) / 0.25 < 0.15

    def test_estimate_depends_on_window_phase_not_just_rate(self):
        # (N-1)/span ignores the partial cycles at the window edges; N/full_window
        # does not, so the scaled estimate of the SAME 0.25 Hz rhythm changes with
        # where the 30 s epoch boundary happens to fall. A rate estimator should
        # depend only on the rhythm, not on the epoch phase.
        ests = []
        for phase in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            t = np.arange(int(30 * FS)) / FS
            x = np.sin(2 * np.pi * 0.25 * t + phase)
            ests.append(rate_peaks_scaled_resp(x, k=1.0, fs=FS))
        assert np.nanstd(ests) > 0.0, (
            "scaled resp estimate is invariant to epoch phase — the definitional "
            "mismatch may have been removed; re-derive respiratory k if so.")


# ── C5: _quality_filter deletes the good beat after an out-of-band gap ─────────

class TestQualityFilterAsymmetry:
    """The ground-truth peak filter removes the peak *after* any out-of-range
    interval and never re-evaluates. A single long gap (missed beat) therefore
    discards a valid downstream beat rather than repairing the gap, biasing the
    reference the whole paper is scored against.
    """

    def test_long_gap_removes_the_good_downstream_peak(self):
        # peaks at 0,100,200, [gap], 450,550,650 ; the 200->450 gap = 2.5 s > 2.0 s max
        peaks = np.array([0, 100, 200, 450, 550, 650])
        kept = _quality_filter(peaks, FS, CARD_LO, CARD_HI)
        assert 450 not in kept, "the good beat after the gap is the one deleted"
        assert 200 in kept, "the beat before the gap is kept"

    def test_does_not_repair_the_gap(self):
        # removing 450 leaves an even longer 200->550 span; the filter made it worse
        peaks = np.array([0, 100, 200, 450, 550, 650])
        kept = _quality_filter(peaks, FS, CARD_LO, CARD_HI).tolist()
        # 450 removed, so consecutive kept beats straddle a >3 s gap
        gaps = np.diff(kept) / FS
        assert gaps.max() > (1.0 / CARD_LO), (
            "the max interval after filtering still violates the band — the "
            "filter deleted a beat without fixing the physiologically impossible "
            "interval it was meant to remove.")

    def test_clean_beats_are_untouched(self):
        peaks = np.arange(0, 3000, 100)  # clean 1 Hz
        kept = _quality_filter(peaks, FS, CARD_LO, CARD_HI)
        assert len(kept) == len(peaks)


# ── C6: meta-test — the existing suite hides the degeneracy by frequency choice ─

def test_existing_suite_frequency_is_the_degenerate_bin():
    """Documents *why* test_rates.py passes on a broken estimator.

    RESP_F0 = 0.25 Hz is the one respiratory frequency at which rate_spectral is
    correct. Any accuracy test of a band estimator must sweep the band, not test
    a single point — see frequency_sweep / the tests above.
    """
    on_bin = rate_spectral(_sine(0.25, 60), RESP_LO, RESP_HI, FS)
    off_bin = rate_spectral(_sine(0.18, 60), RESP_LO, RESP_HI, FS)
    assert abs(on_bin - 0.25) < 1e-6          # the point the old suite tests
    assert abs(off_bin - 0.18) > 0.04         # anywhere else it is wrong
