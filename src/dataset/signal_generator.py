"""
Synthetic Wireless Signal Generator for Automatic Modulation Classification (AMC).
Simulates baseband I/Q samples across multiple digital modulation schemes under
realistic wireless channel impairments (AWGN, Carrier Frequency Offset, Phase Jitter, Rayleigh Fading).
"""

from typing import Tuple, List, Optional, Dict
import numpy as np
import torch


SUPPORTED_MODULATIONS = [
    "BPSK",
    "QPSK",
    "8PSK",
    "16QAM",
    "64QAM",
    "BFSK",
    "CPFSK",
    "PAM4",
]


class WirelessSignalGenerator:
    """Generates complex baseband I/Q signals for digital modulation classification."""

    def __init__(
        self,
        modulations: Optional[List[str]] = None,
        sequence_length: int = 128,
        samples_per_symbol: int = 8,
        seed: Optional[int] = 42,
    ):
        self.modulations = modulations or SUPPORTED_MODULATIONS
        self.sequence_length = sequence_length
        self.sps = samples_per_symbol
        self.num_classes = len(self.modulations)
        self.mod_to_idx: Dict[str, int] = {mod: i for i, mod in enumerate(self.modulations)}
        self.idx_to_mod: Dict[int, str] = {i: mod for i, mod in enumerate(self.modulations)}
        self.rng = np.random.default_rng(seed)

    def _generate_symbols(self, mod_type: str, num_symbols: int) -> np.ndarray:
        """Generate complex constellation symbols for a given modulation scheme."""
        if mod_type == "BPSK":
            bits = self.rng.integers(0, 2, size=num_symbols)
            symbols = 2 * bits - 1.0 + 0j

        elif mod_type == "QPSK":
            bits = self.rng.integers(0, 4, size=num_symbols)
            angles = bits * (np.pi / 2.0) + (np.pi / 4.0)
            symbols = np.exp(1j * angles)

        elif mod_type == "8PSK":
            bits = self.rng.integers(0, 8, size=num_symbols)
            angles = bits * (2.0 * np.pi / 8.0)
            symbols = np.exp(1j * angles)

        elif mod_type == "16QAM":
            # 16-QAM grid: {-3, -1, 1, 3} on real and imag
            i_levels = self.rng.choice([-3.0, -1.0, 1.0, 3.0], size=num_symbols)
            q_levels = self.rng.choice([-3.0, -1.0, 1.0, 3.0], size=num_symbols)
            symbols = (i_levels + 1j * q_levels) / np.sqrt(10.0)

        elif mod_type == "64QAM":
            levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=float)
            i_levels = self.rng.choice(levels, size=num_symbols)
            q_levels = self.rng.choice(levels, size=num_symbols)
            symbols = (i_levels + 1j * q_levels) / np.sqrt(42.0)

        elif mod_type == "BFSK":
            bits = self.rng.integers(0, 2, size=num_symbols)
            freqs = np.where(bits == 0, -1.0, 1.0)
            t = np.arange(num_symbols)
            symbols = np.exp(1j * 2 * np.pi * 0.1 * freqs * t)

        elif mod_type == "CPFSK":
            bits = self.rng.integers(0, 2, size=num_symbols)
            dev = 2 * bits - 1.0
            phase = np.cumsum(np.pi * 0.5 * dev)
            symbols = np.exp(1j * phase)

        elif mod_type == "PAM4":
            bits = self.rng.integers(0, 4, size=num_symbols)
            levels = np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(5.0)
            symbols = levels[bits] + 0j

        else:
            raise ValueError(f"Unsupported modulation type: {mod_type}")

        return symbols

    def _pulse_shape(self, symbols: np.ndarray, target_length: int) -> np.ndarray:
        """Apply simple pulse shaping / oversampling and pulse truncation to target_length."""
        oversampled = np.repeat(symbols, self.sps)
        # Apply raised-cosine or rectangular window
        if len(oversampled) >= target_length:
            return oversampled[:target_length]
        else:
            # Pad with repeating pattern or zeros
            reps = int(np.ceil(target_length / len(oversampled)))
            return np.tile(oversampled, reps)[:target_length]

    def _apply_channel_impairments(
        self,
        signal: np.ndarray,
        snr_db: float,
        cfo_std: float = 0.01,
        phase_jitter_std: float = 0.05,
        rayleigh: bool = False,
    ) -> np.ndarray:
        """Apply AWGN, Carrier Frequency Offset (CFO), phase noise, and Rayleigh fading."""
        n_samples = len(signal)
        t = np.arange(n_samples)

        # 1. Rayleigh multipath fading (flat tap)
        if rayleigh:
            h = (self.rng.normal(0, 1) + 1j * self.rng.normal(0, 1)) / np.sqrt(2.0)
            signal = signal * h

        # 2. Carrier Frequency Offset (CFO) and initial random phase
        cfo = self.rng.normal(0.0, cfo_std)
        phi_0 = self.rng.uniform(0, 2 * np.pi)
        cfo_phase = 2 * np.pi * cfo * t + phi_0
        signal = signal * np.exp(1j * cfo_phase)

        # 3. Phase jitter (Wiener phase noise)
        phase_jitter = np.cumsum(self.rng.normal(0.0, phase_jitter_std, size=n_samples))
        signal = signal * np.exp(1j * phase_jitter)

        # 4. Additive White Gaussian Noise (AWGN)
        sig_power = np.mean(np.abs(signal) ** 2)
        if sig_power <= 0:
            sig_power = 1.0
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = sig_power / snr_linear
        noise = (
            self.rng.normal(0.0, np.sqrt(noise_power / 2.0), size=n_samples)
            + 1j * self.rng.normal(0.0, np.sqrt(noise_power / 2.0), size=n_samples)
        )
        rx_signal = signal + noise

        # Normalize total energy
        rx_power = np.sqrt(np.mean(np.abs(rx_signal) ** 2))
        if rx_power > 0:
            rx_signal = rx_signal / rx_power

        return rx_signal

    def generate_sample(
        self,
        mod_type: str,
        snr_db: float,
        cfo_std: float = 0.005,
        phase_jitter_std: float = 0.02,
        rayleigh: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate a single (I, Q) sample sequence.
        Returns:
            iq_array: ndarray of shape (2, sequence_length), float32.
            label: integer class index.
        """
        num_symbols = int(np.ceil(self.sequence_length / self.sps)) + 4
        symbols = self._generate_symbols(mod_type, num_symbols)
        tx_signal = self._pulse_shape(symbols, self.sequence_length)
        rx_signal = self._apply_channel_impairments(
            tx_signal,
            snr_db=snr_db,
            cfo_std=cfo_std,
            phase_jitter_std=phase_jitter_std,
            rayleigh=rayleigh,
        )

        i_channel = np.real(rx_signal).astype(np.float32)
        q_channel = np.imag(rx_signal).astype(np.float32)
        iq_sample = np.stack([i_channel, q_channel], axis=0)  # Shape: (2, L)

        label = self.mod_to_idx[mod_type]
        return iq_sample, label

    def generate_dataset(
        self,
        samples_per_mod_per_snr: int = 50,
        snr_range: Tuple[int, int, int] = (-10, 20, 4),  # start, stop, step
        cfo_std: float = 0.005,
        phase_jitter_std: float = 0.02,
        rayleigh: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate full dataset across modulations and SNR levels.
        Returns:
            X: ndarray of shape (N, 2, sequence_length)
            y: ndarray of shape (N,)
            snrs: ndarray of shape (N,)
        """
        snr_values = list(range(snr_range[0], snr_range[1] + 1, snr_range[2]))
        total_samples = len(self.modulations) * len(snr_values) * samples_per_mod_per_snr

        X = np.zeros((total_samples, 2, self.sequence_length), dtype=np.float32)
        y = np.zeros((total_samples,), dtype=np.int64)
        snrs = np.zeros((total_samples,), dtype=np.float32)

        idx = 0
        for snr in snr_values:
            for mod in self.modulations:
                for _ in range(samples_per_mod_per_snr):
                    iq_sample, label = self.generate_sample(
                        mod,
                        snr_db=float(snr),
                        cfo_std=cfo_std,
                        phase_jitter_std=phase_jitter_std,
                        rayleigh=rayleigh,
                    )
                    X[idx] = iq_sample
                    y[idx] = label
                    snrs[idx] = float(snr)
                    idx += 1

        # Shuffle
        perm = self.rng.permutation(total_samples)
        return X[perm], y[perm], snrs[perm]
