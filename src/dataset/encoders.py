"""
Neuromorphic Spike Encoders for Wireless Signal I/Q streams.
Converts continuous baseband wireless samples into discrete temporal spike trains (events).
Includes:
- Delta Modulation (Threshold-Crossing UP/DOWN spikes) - Primary event-driven encoder
- Poisson Rate Encoder
- Temporal Latency / Time-to-First-Spike (TTFS) Encoder
"""

from typing import Optional
import torch
import torch.nn as nn
import numpy as np


class DeltaModulationEncoder(nn.Module):
    """
    Delta Modulation (Threshold-Crossing) Spike Encoder.
    Emits an ON (+1) spike when signal increases by more than V_th,
    and an OFF (-1 -> channel 2) spike when signal decreases by more than V_th.

    Input shape: (Batch, Channels=2, SequenceLength) -> I and Q channels.
    Output shape: (SequenceLength, Batch, 4) -> [I_on, I_off, Q_on, Q_off] spike events.
    """

    def __init__(self, threshold: float = 0.15, adaptive: bool = False):
        super().__init__()
        self.threshold = threshold
        self.adaptive = adaptive

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, 2, SequenceLength) representing I and Q.
        Returns:
            spikes: Tensor of shape (SequenceLength, Batch, 4) with binary {0, 1} spikes.
        """
        batch_size, channels, seq_len = x.shape
        assert channels == 2, f"Expected 2 input channels (I, Q), got {channels}"

        # Permute to (seq_len, batch_size, 2)
        x_seq = x.permute(2, 0, 1)

        # Output spike tensor: [I_on, I_off, Q_on, Q_off]
        spikes = torch.zeros((seq_len, batch_size, 4), device=x.device, dtype=torch.float32)

        # Baseline reference initially set to the first sample
        ref = x_seq[0].clone()  # Shape: (batch_size, 2)

        # Compute adaptive threshold if enabled
        if self.adaptive:
            # Per-sample dynamic threshold based on signal variance
            std = torch.std(x, dim=2) + 1e-6  # Shape: (batch, 2)
            v_th = (self.threshold * std).unsqueeze(0)  # Broadcastable
        else:
            v_th = self.threshold

        for t in range(seq_len):
            current = x_seq[t]  # (batch_size, 2)
            diff = current - ref  # (batch_size, 2)

            # Positive crossing (ON)
            on_mask = diff >= (v_th if not self.adaptive else v_th[0])
            # Negative crossing (OFF)
            off_mask = diff <= (-v_th if not self.adaptive else -v_th[0])

            # Channel 0 (I) -> spikes[:, :, 0] (I_on), spikes[:, :, 1] (I_off)
            spikes[t, :, 0] = on_mask[:, 0].float()
            spikes[t, :, 1] = off_mask[:, 0].float()

            # Channel 1 (Q) -> spikes[:, :, 2] (Q_on), spikes[:, :, 3] (Q_off)
            spikes[t, :, 2] = on_mask[:, 1].float()
            spikes[t, :, 3] = off_mask[:, 1].float()

            # Update reference values only where spikes occurred
            fired_mask = on_mask | off_mask
            ref = torch.where(fired_mask, current, ref)

        return spikes


class RateEncoder(nn.Module):
    """
    Poisson Rate Spike Encoder.
    Maps normalized signal amplitude to spike probabilities over T time steps.
    Input shape: (Batch, 2, SequenceLength) or (Batch, 2)
    Output shape: (TimeSteps, Batch, Channels*2)
    """

    def __init__(self, time_steps: int = 16):
        super().__init__()
        self.time_steps = time_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts I/Q amplitudes to Poisson spike trains.
        Separates positive and negative components: [I+, I-, Q+, Q-].
        """
        # Separate positive and negative components
        i_pos = torch.clamp(x[:, 0, :], min=0.0)
        i_neg = torch.clamp(-x[:, 0, :], min=0.0)
        q_pos = torch.clamp(x[:, 1, :], min=0.0)
        q_neg = torch.clamp(-x[:, 1, :], min=0.0)

        # Concatenate 4 channels: (Batch, 4, seq_len)
        features = torch.stack([i_pos, i_neg, q_pos, q_neg], dim=1)

        # Normalize to probability range [0, 0.95]
        max_val = features.max()
        if max_val > 0:
            probs = (features / (max_val + 1e-6)) * 0.9
        else:
            probs = features

        # Time dimension first: (time_steps, batch, 4, seq_len)
        # Flatten spatial channels for feedforward: (time_steps, batch, 4 * seq_len)
        batch_size, _, seq_len = features.shape
        probs_flat = probs.view(batch_size, 4 * seq_len)

        rand = torch.rand((self.time_steps, batch_size, 4 * seq_len), device=x.device)
        spikes = (rand < probs_flat.unsqueeze(0)).float()

        return spikes


class PhaseLatencyEncoder(nn.Module):
    """
    Phase & Latency Encoder for RF Signals.
    Encodes instantaneous phase angle into spike timing (Time-To-First-Spike).
    """

    def __init__(self, time_steps: int = 20):
        super().__init__()
        self.time_steps = time_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch, 2, SequenceLength)
        Returns: (TimeSteps, Batch, 4)
        """
        batch_size, _, seq_len = x.shape
        i_ch = x[:, 0, :]
        q_ch = x[:, 1, :]

        # Mean phase angle over sequence: [-pi, pi]
        phase = torch.atan2(q_ch.mean(dim=1), i_ch.mean(dim=1) + 1e-8)  # (Batch,)
        norm_phase = (phase + np.pi) / (2 * np.pi)  # [0, 1]

        # Latency index: earlier spike for lower phase angle
        spike_time = (norm_phase * (self.time_steps - 1)).long().clamp(0, self.time_steps - 1)

        spikes = torch.zeros((self.time_steps, batch_size, 4), device=x.device)
        for b in range(batch_size):
            t_idx = spike_time[b].item()
            spikes[t_idx, b, :] = 1.0

        return spikes


def get_spike_encoder(encoder_type: str = "delta", **kwargs) -> nn.Module:
    """Factory helper to get the desired neuromorphic spike encoder."""
    enc = encoder_type.lower()
    if enc in ["delta", "threshold", "event"]:
        threshold = kwargs.get("threshold", 0.15)
        adaptive = kwargs.get("adaptive", False)
        return DeltaModulationEncoder(threshold=threshold, adaptive=adaptive)
    elif enc in ["rate", "poisson"]:
        time_steps = kwargs.get("time_steps", 16)
        return RateEncoder(time_steps=time_steps)
    elif enc in ["latency", "ttfs", "phase"]:
        time_steps = kwargs.get("time_steps", 20)
        return PhaseLatencyEncoder(time_steps=time_steps)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
