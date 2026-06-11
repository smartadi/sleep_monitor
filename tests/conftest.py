"""
Shared fixtures for sleep_monitor unit tests.

All fixtures use synthetic signals with known mathematical properties so tests
can assert exact expected values without loading any real data files.
"""

import numpy as np
import pytest

# Match the real config values
FS = 100.0          # Hz
RESP_LO, RESP_HI   = 0.1, 0.5    # respiratory band
CARD_LO, CARD_HI   = 0.5, 3.0    # cardiac band

RESP_F0   = 0.25    # Hz  — well inside resp band, ~15 br/min
CARDIAC_F0 = 1.0    # Hz  — well inside cardiac band, 60 BPM
DURATION  = 60.0    # seconds — long enough for low-frequency resolution


@pytest.fixture(scope="session")
def fs():
    return FS


@pytest.fixture(scope="session")
def resp_bounds():
    return RESP_LO, RESP_HI


@pytest.fixture(scope="session")
def cardiac_bounds():
    return CARD_LO, CARD_HI


@pytest.fixture(scope="session")
def t60():
    """60-second time vector at 100 Hz."""
    return np.arange(int(DURATION * FS)) / FS


@pytest.fixture(scope="session")
def resp_sine(t60):
    """Pure 0.25 Hz sine — clean resp-band signal, known rate."""
    return np.sin(2 * np.pi * RESP_F0 * t60)


@pytest.fixture(scope="session")
def cardiac_sine(t60):
    """Pure 1.0 Hz sine — clean cardiac-band signal, known rate."""
    return np.sin(2 * np.pi * CARDIAC_F0 * t60)


@pytest.fixture(scope="session")
def resp_f0():
    return RESP_F0


@pytest.fixture(scope="session")
def cardiac_f0():
    return CARDIAC_F0
