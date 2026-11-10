"""Unit tests for SNN and ANN models forward pass."""

import torch
import pytest
from src.models import Spiking1DCNN, SpikingMLP, BaselineCNN1D, BaselineGRU, BaselineMLP


def test_spiking_1d_cnn_forward():
    model = Spiking1DCNN(in_channels=4, num_classes=8)
    # Shape: (time_steps=32, batch_size=2, channels=4)
    x = torch.randint(0, 2, (32, 2, 4)).float()
    spk_out, mem_out, layer_spk = model(x, record_all_spikes=True)
    assert spk_out.shape == (32, 2, 8)
    assert mem_out.shape == (32, 2, 8)
    assert "layer1" in layer_spk
    assert "layer2" in layer_spk
    assert "layer3" in layer_spk


def test_spiking_mlp_forward():
    model = SpikingMLP(in_features=4, num_classes=8)
    x = torch.randint(0, 2, (16, 3, 4)).float()
    spk_out, mem_out, _ = model(x)
    assert spk_out.shape == (16, 3, 8)


def test_baseline_ann_models():
    # Shape: (batch=2, channels=2, seq_len=64)
    x = torch.randn(2, 2, 64)

    cnn = BaselineCNN1D(in_channels=2, num_classes=8)
    out_cnn = cnn(x)
    assert out_cnn.shape == (2, 8)

    gru = BaselineGRU(in_features=2, hidden_dim=32, num_classes=8)
    out_gru = gru(x)
    assert out_gru.shape == (2, 8)

    mlp = BaselineMLP(seq_len=64, in_channels=2, num_classes=8)
    out_mlp = mlp(x)
    assert out_mlp.shape == (2, 8)
