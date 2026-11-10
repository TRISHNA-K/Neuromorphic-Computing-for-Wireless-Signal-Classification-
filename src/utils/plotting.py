"""
Visualization and plotting utilities for neuromorphic wireless experiments.
Generates publication-quality figures:
- SNR vs Classification Accuracy curves
- Confusion Matrices
- Energy vs Latency Pareto Frontier
- Neuromorphic Spike Raster Plots
"""

import os
import tempfile
mpl_dir = os.path.join(tempfile.gettempdir(), "mpl_cache")
os.makedirs(mpl_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = mpl_dir

from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def set_plot_style():
    """Apply clean, modern publication plot styling."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 14,
        "lines.linewidth": 2.2,
        "lines.markersize": 7,
    })


def plot_snr_vs_accuracy(
    snr_levels: List[float],
    model_accuracies: Dict[str, List[float]],
    save_path: str,
    title: str = "Modulation Classification Accuracy vs SNR",
):
    """Plot Accuracy vs SNR for SNN and ANN models."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    palette = sns.color_palette("tab10", len(model_accuracies))
    markers = ["o", "s", "^", "D", "v", "<"]

    for i, (name, accs) in enumerate(model_accuracies.items()):
        ax.plot(
            snr_levels,
            [a * 100.0 if max(accs) <= 1.0 else a for a in accs],
            marker=markers[i % len(markers)],
            color=palette[i],
            label=name,
        )

    ax.set_xlabel("Signal-to-Noise Ratio (SNR in dB)")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", frameon=True)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = True,
):
    """Plot normalized confusion matrix heatmap."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    if normalize:
        cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    else:
        cm_norm = cm

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
    )

    ax.set_xlabel("Predicted Modulation")
    ax.set_ylabel("Ground Truth Modulation")
    ax.set_title(title, fontweight="bold")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_energy_vs_latency_pareto(
    comparison_data: List[Dict],
    save_path: str,
    title: str = "Efficiency-Performance Trade-off (Energy Proxy vs Latency)",
):
    """
    Scatter plot comparing energy proxy (pJ) vs latency (ms).
    comparison_data: list of dicts with keys: name, energy_pj, latency_ms, accuracy
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    for item in comparison_data:
        name = item["name"]
        energy = item["energy_pj"]
        latency = item["latency_ms"]
        acc = item.get("accuracy", 80.0)

        # Bubble size proportional to accuracy
        size = (acc - 50.0) * 8 if acc > 50 else 100
        scatter = ax.scatter(
            latency,
            energy,
            s=max(120, size),
            alpha=0.85,
            edgecolors="black",
            linewidth=1.2,
            label=f"{name} ({acc:.1f}%)",
        )
        ax.annotate(
            f" {name}\n ({energy:.0f} pJ)",
            (latency, energy),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9.5,
            fontweight="semibold",
        )

    ax.set_xlabel("Inference Latency (ms)")
    ax.set_ylabel("Estimated Energy Proxy per Sample (pJ) [Log Scale]")
    ax.set_yscale("log")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_spike_raster(
    layer_spikes: Dict[str, np.ndarray],
    save_path: str,
    sample_idx: int = 0,
    title: str = "Neuromorphic Spike Raster (Event Activity across Layers)",
):
    """
    Plot spike raster for input events and hidden SNN layers across time steps.
    layer_spikes: dict of {layer_name: ndarray of shape (T, B, neurons)}
    """
    set_plot_style()
    num_layers = len(layer_spikes)
    fig, axes = plt.subplots(num_layers, 1, figsize=(10, 2.5 * num_layers), sharex=True, dpi=300)

    if num_layers == 1:
        axes = [axes]

    for ax, (lname, spk_arr) in zip(axes, layer_spikes.items()):
        # Shape: (T, B, N) -> extract sample_idx
        sample_spks = spk_arr[:, sample_idx, :]  # (T, N)
        t_indices, n_indices = np.where(sample_spks > 0)

        ax.scatter(t_indices, n_indices, color="navy", marker="|", s=40, alpha=0.8)
        ax.set_ylabel(f"{lname}\nNeuron ID")
        ax.set_ylim(-0.5, sample_spks.shape[1] - 0.5)
        ax.grid(True, linestyle=":", alpha=0.4)

    axes[-1].set_xlabel("Simulation Time Step (t)")
    fig.suptitle(title, fontweight="bold", y=0.99)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
