"""Unit tests for computational activity and energy meter."""

import torch
import pytest
from src.models import Spiking1DCNN, BaselineCNN1D
from src.metrics.energy_meter import EnergyMeter


def test_ann_flops_and_macs():
    model = BaselineCNN1D(in_channels=2, num_classes=8)
    metrics = EnergyMeter.calculate_ann_macs_and_flops(model, (1, 2, 128))
    assert metrics["macs_per_sample"] > 0
    assert metrics["flops_per_sample"] == metrics["macs_per_sample"] * 2.0
    assert metrics["energy_proxy_pj"] > 0


def test_snn_sops_and_energy():
    model = Spiking1DCNN(in_channels=4, num_classes=8)
    spikes = (torch.rand(32, 2, 4) > 0.8).float()
    metrics = EnergyMeter.profile_snn_activity(model, spikes)
    assert metrics["sops_per_sample"] >= 0
    assert 0.0 <= metrics["spike_sparsity_percent"] <= 100.0
    assert metrics["energy_proxy_pj"] >= 0


def test_compare_efficiency():
    ann_m = {"macs_per_sample": 100000, "energy_proxy_pj": 460000}
    snn_m = {"sops_per_sample": 10000, "energy_proxy_pj": 9000}
    comp = EnergyMeter.compare_efficiency(ann_m, snn_m)
    assert comp["energy_efficiency_gain"] > 1.0
    assert comp["energy_saved_percent"] > 0.0
