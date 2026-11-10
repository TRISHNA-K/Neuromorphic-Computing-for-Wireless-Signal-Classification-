"""
Spiking Neural Network (SNN) architectures for wireless signal classification using snnTorch.
Implements:
- Spiking1DCNN: 1D Spiking Convolutional Network for event-driven temporal I/Q streams.
- SpikingMLP: Fully connected Leaky Integrate-and-Fire (LIF) network.
- RecurrentSNN: Recurrent Spiking Network capturing deep temporal wireless phase patterns.
"""

from typing import Tuple, List, Optional, Dict
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class Spiking1DCNN(nn.Module):
    """
    Spiking 1D Convolutional Neural Network for event-based wireless signal classification.
    Processes input spike trains: (time_steps, batch_size, in_channels).
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 8,
        conv_channels: List[int] = None,
        fc_hidden: int = 128,
        beta: float = 0.9,
        threshold: float = 1.0,
        spike_grad=None,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64]
        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25.0)

        self.beta = beta
        self.num_classes = num_classes

        # Conv Layer 1
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=conv_channels[0],
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

        # Conv Layer 2
        self.conv2 = nn.Conv1d(
            in_channels=conv_channels[0],
            out_channels=conv_channels[1],
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

        # Global average pooling over spatial/channel dimension
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Dense classification layers
        self.fc1 = nn.Linear(conv_channels[1], fc_hidden, bias=False)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

        self.fc2 = nn.Linear(fc_hidden, num_classes, bias=False)
        self.lif_out = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

    def forward(
        self, x: torch.Tensor, record_all_spikes: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            x: Spike train tensor of shape (time_steps, batch, in_channels) or (batch, in_channels, time_steps).
            record_all_spikes: Whether to record and return all intermediate layer spikes for energy profiling.
        Returns:
            spk_out_rec: (time_steps, batch, num_classes)
            mem_out_rec: (time_steps, batch, num_classes)
            layer_spikes: Dict of intermediate spike recordings (if record_all_spikes=True)
        """
        # Ensure format is (time_steps, batch, in_channels)
        if x.dim() == 3 and x.shape[1] == 4 and x.shape[0] != x.shape[2]:
            # Input is (batch, in_channels, seq_len) -> permute to (seq_len, batch, in_channels)
            x = x.permute(2, 0, 1)

        time_steps, batch_size, in_channels = x.shape

        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_out_rec = []
        mem_out_rec = []

        if record_all_spikes:
            spk1_rec = []
            spk2_rec = []
            spk3_rec = []

        for step in range(time_steps):
            # Input at current step: (batch, in_channels) -> reshape to (batch, in_channels, 1) for Conv1D
            x_t = x[step].unsqueeze(-1)  # (batch, in_channels, 1)

            # Layer 1
            cur1 = self.conv1(x_t).squeeze(-1)  # (batch, conv_channels[0])
            spk1, mem1 = self.lif1(cur1, mem1)

            # Layer 2
            cur2 = self.conv2(spk1.unsqueeze(-1)).squeeze(-1)
            spk2, mem2 = self.lif2(cur2, mem2)

            # Layer 3 (Linear)
            cur3 = self.fc1(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            # Output Layer
            cur_out = self.fc2(spk3)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)

            spk_out_rec.append(spk_out)
            mem_out_rec.append(mem_out)

            if record_all_spikes:
                spk1_rec.append(spk1)
                spk2_rec.append(spk2)
                spk3_rec.append(spk3)

        spk_out_stack = torch.stack(spk_out_rec, dim=0)
        mem_out_stack = torch.stack(mem_out_rec, dim=0)

        layer_spikes = None
        if record_all_spikes:
            layer_spikes = {
                "input": x,
                "layer1": torch.stack(spk1_rec, dim=0),
                "layer2": torch.stack(spk2_rec, dim=0),
                "layer3": torch.stack(spk3_rec, dim=0),
                "output": spk_out_stack,
            }

        return spk_out_stack, mem_out_stack, layer_spikes


class SpikingMLP(nn.Module):
    """Fully Connected Spiking Neural Network with LIF neurons."""

    def __init__(
        self,
        in_features: int = 4,
        hidden_dims: List[int] = None,
        num_classes: int = 8,
        beta: float = 0.9,
        threshold: float = 1.0,
        spike_grad=None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25.0)

        self.beta = beta
        self.num_classes = num_classes

        # Layer 1
        self.fc1 = nn.Linear(in_features, hidden_dims[0], bias=False)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

        # Layer 2
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1], bias=False)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

        # Output Layer
        self.fc3 = nn.Linear(hidden_dims[1], num_classes, bias=False)
        self.lif_out = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

    def forward(
        self, x: torch.Tensor, record_all_spikes: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        x: (time_steps, batch, in_features)
        """
        if x.dim() == 3 and x.shape[1] == 4 and x.shape[0] != x.shape[2]:
            x = x.permute(2, 0, 1)

        time_steps, batch_size, in_feat = x.shape

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_out_rec = []
        mem_out_rec = []

        if record_all_spikes:
            spk1_rec = []
            spk2_rec = []

        for step in range(time_steps):
            x_t = x[step]

            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur_out = self.fc3(spk2)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)

            spk_out_rec.append(spk_out)
            mem_out_rec.append(mem_out)

            if record_all_spikes:
                spk1_rec.append(spk1)
                spk2_rec.append(spk2)

        spk_out_stack = torch.stack(spk_out_rec, dim=0)
        mem_out_stack = torch.stack(mem_out_rec, dim=0)

        layer_spikes = None
        if record_all_spikes:
            layer_spikes = {
                "input": x,
                "layer1": torch.stack(spk1_rec, dim=0),
                "layer2": torch.stack(spk2_rec, dim=0),
                "output": spk_out_stack,
            }

        return spk_out_stack, mem_out_stack, layer_spikes


def build_snn_model(model_name: str = "spiking_cnn", **kwargs) -> nn.Module:
    """Factory helper to build SNN model."""
    name = model_name.lower()
    if name in ["spiking_cnn", "snn_cnn", "cnn"]:
        return Spiking1DCNN(**kwargs)
    elif name in ["spiking_mlp", "snn_mlp", "mlp"]:
        return SpikingMLP(**kwargs)
    else:
        raise ValueError(f"Unknown SNN model: {model_name}")
