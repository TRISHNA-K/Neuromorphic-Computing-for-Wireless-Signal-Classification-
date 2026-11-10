"""
Benchmarking script for Computational Activity, Latency, and Estimated Energy Proxies (SNN vs ANN).
"""

import os
import argparse
from typing import Dict, Any
import torch

from src.dataset import WirelessSignalGenerator, WirelessSignalDataset
from src.models import build_snn_model, build_ann_model
from src.metrics import EnergyMeter, LatencyMeter
from src.utils import set_seed, save_json, get_device


def benchmark_snn_vs_ann(
    snn_checkpoint: str,
    ann_checkpoint: str,
    output_dir: str = "results",
    seed: int = 42,
) -> Dict[str, Any]:
    """Run full computational activity, latency, and energy benchmark."""
    set_seed(seed)
    device = get_device()
    print(f"[+] Benchmarking on device: {device}")

    # Load SNN
    snn_ckpt = torch.load(snn_checkpoint, map_location=device, weights_only=False)
    snn_arch = snn_ckpt.get("arch", "spiking_cnn")
    snn_encoder = snn_ckpt.get("encoder", "delta")
    num_classes = snn_ckpt.get("meta", {}).get("num_classes", 8)

    snn_model = build_snn_model(snn_arch, in_channels=4, num_classes=num_classes).to(device)
    snn_model.load_state_dict(snn_ckpt["model_state_dict"])
    snn_model.eval()

    # Load ANN
    ann_ckpt = torch.load(ann_checkpoint, map_location=device, weights_only=False)
    ann_arch = ann_ckpt.get("arch", "cnn1d")
    ann_model = build_ann_model(ann_arch, in_channels=2, num_classes=num_classes).to(device)
    ann_model.load_state_dict(ann_ckpt["model_state_dict"])
    ann_model.eval()

    # Generate benchmark test batch
    gen = WirelessSignalGenerator(sequence_length=128, seed=seed)
    X, y, snrs = gen.generate_dataset(samples_per_mod_per_snr=10, snr_range=(10, 10, 1))

    # Continuous input for ANN: shape (B, 2, 128)
    ann_inputs = torch.from_numpy(X).float().to(device)

    # Spike input for SNN: shape (128, B, 4)
    snn_dataset = WirelessSignalDataset(X, y, snrs, encoder_type=snn_encoder)
    # Collect all spikes
    snn_spikes_list = [snn_dataset[i][0] for i in range(len(snn_dataset))]
    # Each item is (128, 4) -> stack to (B, 128, 4) -> permute to (128, B, 4)
    snn_spikes = torch.stack(snn_spikes_list, dim=0).permute(1, 0, 2).to(device)

    print("\n--- 1. COMPUTATIONAL ACTIVITY & ENERGY PROFILES ---")
    ann_activity = EnergyMeter.calculate_ann_macs_and_flops(ann_model, input_shape=(1, 2, 128))
    snn_activity = EnergyMeter.profile_snn_activity(snn_model, snn_spikes, device=device)
    efficiency = EnergyMeter.compare_efficiency(ann_activity, snn_activity)

    print(f"ANN MACs / Sample:            {ann_activity['macs_per_sample']:,.0f}")
    print(f"SNN SOPs / Sample:            {snn_activity['sops_per_sample']:,.0f}")
    print(f"SNN Spike Sparsity:           {snn_activity['spike_sparsity_percent']:.2f}%")
    print(f"SNN Activity Factor:          {snn_activity['activity_factor']:.4f}")
    print(f"ANN Energy Proxy:             {ann_activity['energy_proxy_pj']:.1f} pJ")
    print(f"SNN Energy Proxy:             {snn_activity['energy_proxy_pj']:.1f} pJ")
    print(f"Energy Efficiency Gain:       {efficiency['energy_efficiency_gain']:.2f}x")
    print(f"Energy Saved:                 {efficiency['energy_saved_percent']:.2f}%")

    print("\n--- 2. INFERENCE LATENCY & THROUGHPUT ---")
    single_ann = ann_inputs[:1]
    single_snn = snn_spikes[:, :1, :]

    ann_lat = LatencyMeter.measure_latency(ann_model, single_ann, num_warmup=25, num_runs=150, device=device)
    snn_lat = LatencyMeter.measure_latency(snn_model, single_snn, num_warmup=25, num_runs=150, device=device)

    print(f"ANN Latency (mean ± std):     {ann_lat['mean_latency_ms']:.3f} ± {ann_lat['std_latency_ms']:.3f} ms")
    print(f"ANN Latency (p95):            {ann_lat['p95_latency_ms']:.3f} ms")
    print(f"ANN Throughput:               {ann_lat['throughput_samples_per_sec']:.1f} samples/s")

    print(f"SNN Latency (mean ± std):     {snn_lat['mean_latency_ms']:.3f} ± {snn_lat['std_latency_ms']:.3f} ms")
    print(f"SNN Latency (p95):            {snn_lat['p95_latency_ms']:.3f} ms")
    print(f"SNN Throughput:               {snn_lat['throughput_samples_per_sec']:.1f} samples/s")

    print("\n--- 3. SNN EARLY-DECISION TIME-TO-CLASSIFICATION ---")
    early_decision = LatencyMeter.evaluate_early_decision_snn(snn_model, snn_spikes, confidence_threshold=0.85)
    print(f"Total Sequence Length:        {early_decision['total_time_steps']} steps")
    print(f"Avg Steps to Decision:        {early_decision['avg_steps_to_decision']:.1f} steps")
    print(f"Fraction of Sequence Used:    {early_decision['fraction_of_sequence_used'] * 100:.1f}%")
    print(f"Temporal Speedup Potential:   {early_decision['temporal_speedup_potential']:.2f}x")

    benchmark_summary = {
        "models": {
            "snn": {"arch": snn_arch, "encoder": snn_encoder},
            "ann": {"arch": ann_arch},
        },
        "computational_activity": {
            "ann": ann_activity,
            "snn": snn_activity,
            "comparison": efficiency,
        },
        "latency": {
            "ann": ann_lat,
            "snn": snn_lat,
        },
        "snn_early_decision": early_decision,
    }

    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "energy_and_latency_benchmark.json")
    save_json(benchmark_summary, summary_path)
    print(f"\n[+] Saved complete benchmark results to {summary_path}")

    return benchmark_summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark energy proxy and latency for SNN vs ANN.")
    parser.add_argument("--snn_checkpoint", type=str, default="checkpoints/snn_model.pth")
    parser.add_argument("--ann_checkpoint", type=str, default="checkpoints/ann_model.pth")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    benchmark_snn_vs_ann(
        snn_checkpoint=args.snn_checkpoint,
        ann_checkpoint=args.ann_checkpoint,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
