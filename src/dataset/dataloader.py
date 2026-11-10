"""
PyTorch Dataset and DataLoader abstractions for wireless signal classification.
Supports both continuous I/Q baseband formats (for ANNs) and event-driven spike trains (for SNNs).
"""

from typing import Tuple, Optional, Dict
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

from .signal_generator import WirelessSignalGenerator, SUPPORTED_MODULATIONS
from .encoders import get_spike_encoder


class WirelessSignalDataset(Dataset):
    """
    Dataset wrapping wireless I/Q signals.
    Optionally applies an on-the-fly spike encoder for SNN pipelines.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        snrs: np.ndarray,
        encoder_type: Optional[str] = None,
        encoder_kwargs: Optional[Dict] = None,
    ):
        """
        Args:
            X: Array of shape (N, 2, seq_len), float32.
            y: Array of shape (N,), int64.
            snrs: Array of shape (N,), float32.
            encoder_type: Optional spike encoder name ('delta', 'rate', 'latency').
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.snrs = torch.from_numpy(snrs).float()
        self.encoder_type = encoder_type

        if encoder_type:
            kwargs = encoder_kwargs or {}
            self.encoder = get_spike_encoder(encoder_type, **kwargs)
            # Batch encode entire dataset at once for ultra-fast vectorization
            with torch.no_grad():
                # self.encoder(self.X) returns (seq_len, N, channels) -> permute to (N, seq_len, channels)
                self.spikes = self.encoder(self.X).permute(1, 0, 2)
        else:
            self.encoder = None
            self.spikes = None

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        label = self.y[idx]
        snr = self.snrs[idx].item()

        if self.spikes is not None:
            return self.spikes[idx], label, snr

        return self.X[idx], label, snr


def get_wireless_dataloaders(
    samples_per_mod_per_snr: int = 50,
    snr_range: Tuple[int, int, int] = (-10, 20, 4),
    sequence_length: int = 128,
    batch_size: int = 64,
    train_split: float = 0.7,
    val_split: float = 0.15,
    encoder_type: Optional[str] = None,
    encoder_kwargs: Optional[Dict] = None,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Generate synthetic wireless dataset and construct Train, Validation, and Test DataLoaders.
    Returns:
        train_loader, val_loader, test_loader, metadata_dict
    """
    gen = WirelessSignalGenerator(
        sequence_length=sequence_length,
        seed=seed,
    )
    X, y, snrs = gen.generate_dataset(
        samples_per_mod_per_snr=samples_per_mod_per_snr,
        snr_range=snr_range,
    )

    full_dataset = WirelessSignalDataset(
        X=X,
        y=y,
        snrs=snrs,
        encoder_type=encoder_type,
        encoder_kwargs=encoder_kwargs,
    )

    total_len = len(full_dataset)
    train_len = int(train_split * total_len)
    val_len = int(val_split * total_len)
    test_len = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_len, val_len, test_len], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    meta = {
        "num_classes": gen.num_classes,
        "classes": gen.modulations,
        "sequence_length": sequence_length,
        "total_samples": total_len,
        "train_samples": train_len,
        "val_samples": val_len,
        "test_samples": test_len,
        "encoder_type": encoder_type,
    }

    return train_loader, val_loader, test_loader, meta
