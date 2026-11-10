"""
Conventional Artificial Neural Network (ANN) baseline models for wireless signal classification.
- BaselineCNN1D: Standard 1D Convolutional Neural Network (RadioML style).
- BaselineGRU: Recurrent baseline modeling temporal I/Q dynamics.
- BaselineMLP: Feedforward baseline.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn


class BaselineCNN1D(nn.Module):
    """
    Conventional 1D CNN baseline for modulation classification.
    Input: (batch_size, 2, sequence_length)
    Output: (batch_size, num_classes)
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 8,
        conv_channels: Optional[List[int]] = None,
        fc_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [64, 128]

        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels[1]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(conv_channels[1], fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x shape: (batch_size, 2, seq_len)"""
        feat = self.features(x)
        out = self.classifier(feat)
        return out


class BaselineGRU(nn.Module):
    """
    Conventional GRU baseline for sequential I/Q modulation classification.
    Input: (batch_size, 2, sequence_length)
    Output: (batch_size, num_classes)
    """

    def __init__(
        self,
        in_features: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (batch_size, 2, seq_len) -> permute to (batch_size, seq_len, 2)"""
        x_seq = x.permute(0, 2, 1)
        out, _ = self.gru(x_seq)
        # Use final time step hidden representation
        final_step = out[:, -1, :]
        return self.fc(final_step)


class BaselineMLP(nn.Module):
    """
    Conventional MLP baseline flattening the I/Q sequence.
    Input: (batch_size, 2, sequence_length)
    """

    def __init__(
        self,
        seq_len: int = 128,
        in_channels: int = 2,
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 8,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        in_feat = seq_len * in_channels
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_feat, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_ann_model(model_name: str = "cnn1d", **kwargs) -> nn.Module:
    """Factory helper to build conventional ANN baseline."""
    name = model_name.lower()
    if name in ["cnn", "cnn1d", "baseline_cnn"]:
        return BaselineCNN1D(**kwargs)
    elif name in ["gru", "rnn", "baseline_gru"]:
        return BaselineGRU(**kwargs)
    elif name in ["mlp", "baseline_mlp"]:
        return BaselineMLP(**kwargs)
    else:
        raise ValueError(f"Unknown ANN model: {model_name}")
