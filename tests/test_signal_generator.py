"""Unit tests for wireless signal generation."""

import numpy as np
import pytest
from src.dataset.signal_generator import WirelessSignalGenerator, SUPPORTED_MODULATIONS


def test_signal_generator_initialization():
    gen = WirelessSignalGenerator(sequence_length=128)
    assert gen.num_classes == len(SUPPORTED_MODULATIONS)
    assert gen.sequence_length == 128


def test_generate_sample():
    gen = WirelessSignalGenerator(sequence_length=128, seed=42)
    for mod in SUPPORTED_MODULATIONS:
        iq_sample, label = gen.generate_sample(mod, snr_db=10.0)
        assert iq_sample.shape == (2, 128)
        assert isinstance(label, int)
        assert 0 <= label < len(SUPPORTED_MODULATIONS)
        assert not np.isnan(iq_sample).any()
        assert not np.isinf(iq_sample).any()


def test_generate_dataset():
    gen = WirelessSignalGenerator(sequence_length=64, seed=42)
    X, y, snrs = gen.generate_dataset(samples_per_mod_per_snr=2, snr_range=(0, 4, 4))
    assert X.shape == (2 * len(SUPPORTED_MODULATIONS) * 2, 2, 64)
    assert len(y) == len(X)
    assert len(snrs) == len(X)
