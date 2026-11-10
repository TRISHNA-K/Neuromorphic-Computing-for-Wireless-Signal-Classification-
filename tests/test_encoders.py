"""Unit tests for neuromorphic spike encoders."""

import torch
import pytest
from src.dataset.encoders import DeltaModulationEncoder, RateEncoder, PhaseLatencyEncoder, get_spike_encoder


def test_delta_modulation_encoder():
    encoder = DeltaModulationEncoder(threshold=0.1)
    # Batch size 4, 2 channels (I, Q), sequence length 32
    x = torch.randn(4, 2, 32)
    spikes = encoder(x)
    assert spikes.shape == (32, 4, 4)
    # Check spikes are binary
    unique_vals = torch.unique(spikes).tolist()
    for val in unique_vals:
        assert val in [0.0, 1.0]


def test_rate_encoder():
    encoder = RateEncoder(time_steps=10)
    x = torch.randn(4, 2, 32)
    spikes = encoder(x)
    assert spikes.shape == (10, 4, 4 * 32)
    unique_vals = torch.unique(spikes).tolist()
    for val in unique_vals:
        assert val in [0.0, 1.0]


def test_phase_latency_encoder():
    encoder = PhaseLatencyEncoder(time_steps=16)
    x = torch.randn(4, 2, 32)
    spikes = encoder(x)
    assert spikes.shape == (16, 4, 4)


def test_get_spike_encoder_factory():
    enc = get_spike_encoder("delta", threshold=0.2)
    assert isinstance(enc, DeltaModulationEncoder)
    assert enc.threshold == 0.2
