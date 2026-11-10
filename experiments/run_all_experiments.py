"""
Master Experiment Runner:
Executes the full pipeline for Neuromorphic Computing for Wireless Signal Classification:
1. Trains SNN (Spiking1DCNN with LIF neurons) and ANN baseline (BaselineCNN1D).
2. Evaluates across full SNR sweep (-10 dB to +20 dB).
3. Profiles computational activity, synaptic operations (SOPs), FLOPs, and energy proxies.
4. Generates publication-ready figures in results/figures/.
"""

import os
import sys
import tempfile
mpl_dir = os.path.join(tempfile.gettempdir(), "mpl_cache")
os.makedirs(mpl_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = mpl_dir
import torch
import numpy as np

# Ensure root directory is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.dataset import WirelessSignalGenerator, WirelessSignalDataset, get_wireless_dataloaders, SUPPORTED_MODULATIONS
from src.models import build_snn_model, build_ann_model
from src.metrics import EnergyMeter, LatencyMeter
from src.utils import (
    set_seed,
    save_json,
    plot_snr_vs_accuracy,
    plot_confusion_matrix,
    plot_energy_vs_latency_pareto,
    plot_spike_raster,
    get_device,
)
from experiments.train import train_snn_epoch, eval_snn, train_ann_epoch, eval_ann
from experiments.evaluate import evaluate_snr_sweep


def run_full_suite():
    set_seed(42)
    device = get_device()
    print("=" * 75)
    print("NEUROMORPHIC COMPUTING FOR WIRELESS SIGNAL CLASSIFICATION")
    print(f"Device: {device} | Python & PyTorch + snnTorch Event-Driven Pipeline")
    print("=" * 75)

    checkpoints_dir = os.path.join(root_dir, "checkpoints")
    results_dir = os.path.join(root_dir, "results")
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Dataset Generation
    print("\n[Step 1/5] Generating Synthetic Wireless Signal Datasets...")
    samples_per_mod = 50
    seq_len = 128
    train_loader_ann, val_loader_ann, test_loader_ann, meta = get_wireless_dataloaders(
        samples_per_mod_per_snr=samples_per_mod,
        sequence_length=seq_len,
        batch_size=64,
        encoder_type=None,
        seed=42,
    )
    train_loader_snn, val_loader_snn, test_loader_snn, _ = get_wireless_dataloaders(
        samples_per_mod_per_snr=samples_per_mod,
        sequence_length=seq_len,
        batch_size=64,
        encoder_type="delta",
        seed=42,
    )
    print(f"Dataset successfully created with {meta['num_classes']} classes: {meta['classes']}")
    print(f"Train samples: {meta['train_samples']} | Test samples: {meta['test_samples']}")

    # 2. Train ANN Baseline (or load existing)
    ann_ckpt_path = os.path.join(checkpoints_dir, "ann_model.pth")
    ann_model = build_ann_model("cnn1d", in_channels=2, num_classes=meta["num_classes"]).to(device)
    ann_crit = torch.nn.CrossEntropyLoss()

    if os.path.exists(ann_ckpt_path):
        print(f"\n[Step 2/5] Loading existing trained Baseline CNN-1D checkpoint from {ann_ckpt_path}...")
        ann_ckpt = torch.load(ann_ckpt_path, map_location=device, weights_only=False)
        ann_model.load_state_dict(ann_ckpt["model_state_dict"])
    else:
        print("\n[Step 2/5] Training Baseline CNN-1D (Continuous Baseband)...")
        ann_opt = torch.optim.AdamW(ann_model.parameters(), lr=2e-3, weight_decay=1e-4)
        best_ann_acc = 0.0
        for epoch in range(1, 13):
            tr = train_ann_epoch(ann_model, train_loader_ann, ann_opt, ann_crit, device)
            val = eval_ann(ann_model, val_loader_ann, ann_crit, device)
            if val["accuracy"] > best_ann_acc:
                best_ann_acc = val["accuracy"]
                torch.save(
                    {
                        "model_state_dict": ann_model.state_dict(),
                        "model_type": "ann",
                        "arch": "cnn1d",
                        "meta": meta,
                    },
                    ann_ckpt_path,
                )
            if epoch % 3 == 0 or epoch == 12:
                print(f"  ANN Epoch {epoch:02d} -> Train Loss: {tr['loss']:.3f} | Val Acc: {val['accuracy']:.2f}%")
        ann_ckpt = torch.load(ann_ckpt_path, map_location=device, weights_only=False)
        ann_model.load_state_dict(ann_ckpt["model_state_dict"])

    ann_test_res = eval_ann(ann_model, test_loader_ann, ann_crit, device)
    print(f"[+] ANN Test Accuracy: {ann_test_res['accuracy']:.2f}%")

    # 3. Train SNN Model (or load existing)
    snn_ckpt_path = os.path.join(checkpoints_dir, "snn_model.pth")
    snn_model = build_snn_model("spiking_cnn", in_channels=4, num_classes=meta["num_classes"]).to(device)

    if os.path.exists(snn_ckpt_path):
        print(f"\n[Step 3/5] Loading existing trained Spiking1DCNN checkpoint from {snn_ckpt_path}...")
        snn_ckpt = torch.load(snn_ckpt_path, map_location=device, weights_only=False)
        snn_model.load_state_dict(snn_ckpt["model_state_dict"])
    else:
        print("\n[Step 3/5] Training Spiking Neural Network (Spiking1DCNN with LIF Neurons)...")
        snn_opt = torch.optim.Adam(snn_model.parameters(), lr=3e-3, weight_decay=1e-4)
        best_snn_acc = 0.0
        for epoch in range(1, 15):
            tr = train_snn_epoch(snn_model, train_loader_snn, snn_opt, device)
            val = eval_snn(snn_model, val_loader_snn, device)
            if val["accuracy"] > best_snn_acc:
                best_snn_acc = val["accuracy"]
                torch.save(
                    {
                        "model_state_dict": snn_model.state_dict(),
                        "model_type": "snn",
                        "arch": "spiking_cnn",
                        "encoder": "delta",
                        "meta": meta,
                    },
                    snn_ckpt_path,
                )
            if epoch % 3 == 0 or epoch == 14:
                print(f"  SNN Epoch {epoch:02d} -> Train Loss: {tr['loss']:.3f} | Val Acc: {val['accuracy']:.2f}%")
        snn_ckpt = torch.load(snn_ckpt_path, map_location=device, weights_only=False)
        snn_model.load_state_dict(snn_ckpt["model_state_dict"])

    snn_test_res = eval_snn(snn_model, test_loader_snn, device)
    print(f"[+] SNN Test Accuracy: {snn_test_res['accuracy']:.2f}%")

    # 4. Comprehensive Evaluation across SNR Sweep
    print("\n[Step 4/5] Evaluating Classification Accuracy across SNR Sweep (-10 dB to +20 dB)...")
    snr_levels = [-10, -6, -2, 2, 6, 10, 14, 18, 20]
    ann_snr_res = evaluate_snr_sweep(ann_model, "ann", snr_levels=snr_levels, samples_per_snr=25, device=device)
    snn_snr_res = evaluate_snr_sweep(snn_model, "snn", encoder_type="delta", snr_levels=snr_levels, samples_per_snr=25, device=device)

    save_json(ann_snr_res, os.path.join(results_dir, "ann_snr_results.json"))
    save_json(snn_snr_res, os.path.join(results_dir, "snn_snr_results.json"))

    # Plot SNR vs Accuracy
    plot_snr_vs_accuracy(
        snr_levels=snr_levels,
        model_accuracies={
            "Conventional ANN (1D-CNN)": ann_snr_res["accuracies"],
            "Neuromorphic SNN (Spiking-CNN)": snn_snr_res["accuracies"],
        },
        save_path=os.path.join(figures_dir, "snr_vs_accuracy.png"),
        title="Modulation Classification Accuracy vs SNR (AWGN + CFO + Fading)",
    )

    # Plot Confusion Matrices
    plot_confusion_matrix(
        np.array(ann_snr_res["confusion_matrix"]),
        SUPPORTED_MODULATIONS,
        save_path=os.path.join(figures_dir, "confusion_matrix_ann.png"),
        title="Confusion Matrix - Conventional ANN Baseline",
    )
    plot_confusion_matrix(
        np.array(snn_snr_res["confusion_matrix"]),
        SUPPORTED_MODULATIONS,
        save_path=os.path.join(figures_dir, "confusion_matrix_snn.png"),
        title="Confusion Matrix - Neuromorphic SNN Pipeline",
    )

    # 5. Computational Activity, Synaptic Operations, and Energy Benchmarking
    print("\n[Step 5/5] Profiling Computational Activity, SOPs, Latency & Energy Proxies...")
    gen = WirelessSignalGenerator(sequence_length=seq_len, seed=99)
    X_test, y_test, snrs_test = gen.generate_dataset(samples_per_mod_per_snr=15, snr_range=(10, 10, 1))

    ann_sample = torch.from_numpy(X_test).float().to(device)
    snn_ds = WirelessSignalDataset(X_test, y_test, snrs_test, encoder_type="delta")
    snn_spikes_list = [snn_ds[i][0] for i in range(len(snn_ds))]
    snn_sample = torch.stack(snn_spikes_list, dim=0).permute(1, 0, 2).to(device)

    ann_ops = EnergyMeter.calculate_ann_macs_and_flops(ann_model, input_shape=(1, 2, seq_len))
    snn_ops = EnergyMeter.profile_snn_activity(snn_model, snn_sample, device=device)
    efficiency = EnergyMeter.compare_efficiency(ann_ops, snn_ops)

    ann_lat = LatencyMeter.measure_latency(ann_model, ann_sample[:1], num_warmup=15, num_runs=50, device=device)
    snn_lat = LatencyMeter.measure_latency(snn_model, snn_sample[:, :1, :], num_warmup=15, num_runs=50, device=device)
    early_decision = LatencyMeter.evaluate_early_decision_snn(snn_model, snn_sample)

    # Plot Energy vs Latency Pareto
    comparison_data = [
        {
            "name": "Conventional ANN (1D-CNN)",
            "energy_pj": ann_ops["energy_proxy_pj"],
            "latency_ms": ann_lat["mean_latency_ms"],
            "accuracy": ann_snr_res["overall_accuracy"] * 100.0,
        },
        {
            "name": "Neuromorphic SNN (Spiking-CNN)",
            "energy_pj": snn_ops["energy_proxy_pj"],
            "latency_ms": snn_lat["mean_latency_ms"],
            "accuracy": snn_snr_res["overall_accuracy"] * 100.0,
        },
    ]
    plot_energy_vs_latency_pareto(
        comparison_data,
        save_path=os.path.join(figures_dir, "energy_vs_latency_pareto.png"),
        title="Neuromorphic Energy-Latency Frontier (45nm CMOS Energy Proxy)",
    )

    # Plot Spike Raster
    with torch.no_grad():
        _, _, layer_spikes = snn_model(snn_sample[:, :1, :], record_all_spikes=True)
    layer_spikes_np = {k: v.cpu().numpy() for k, v in layer_spikes.items() if k != "input"}
    plot_spike_raster(
        layer_spikes_np,
        save_path=os.path.join(figures_dir, "spike_raster_sample.png"),
        sample_idx=0,
        title="Event-Driven Spike Raster across SNN Layers",
    )

    # Compile Summary
    summary = {
        "classification_performance": {
            "snr_levels": snr_levels,
            "ann_accuracy_by_snr": ann_snr_res["accuracies"],
            "snn_accuracy_by_snr": snn_snr_res["accuracies"],
            "ann_overall_accuracy": ann_snr_res["overall_accuracy"],
            "snn_overall_accuracy": snn_snr_res["overall_accuracy"],
        },
        "computational_activity_and_energy": {
            "ann_macs_per_sample": ann_ops["macs_per_sample"],
            "snn_sops_per_sample": snn_ops["sops_per_sample"],
            "snn_spike_sparsity_percent": snn_ops["spike_sparsity_percent"],
            "snn_activity_factor": snn_ops["activity_factor"],
            "ann_energy_proxy_pj": ann_ops["energy_proxy_pj"],
            "snn_energy_proxy_pj": snn_ops["energy_proxy_pj"],
            "energy_efficiency_gain_factor": efficiency["energy_efficiency_gain"],
            "energy_saved_percent": efficiency["energy_saved_percent"],
        },
        "latency_and_throughput": {
            "ann_latency_ms": ann_lat["mean_latency_ms"],
            "snn_latency_ms": snn_lat["mean_latency_ms"],
            "ann_throughput_samples_per_sec": ann_lat["throughput_samples_per_sec"],
            "snn_throughput_samples_per_sec": snn_lat["throughput_samples_per_sec"],
            "snn_early_decision_avg_steps": early_decision["avg_steps_to_decision"],
            "snn_temporal_speedup_factor": early_decision["temporal_speedup_potential"],
        },
    }

    summary_file = os.path.join(results_dir, "benchmark_summary.json")
    save_json(summary, summary_file)

    print("\n" + "=" * 75)
    print("EXPERIMENT SUITE COMPLETED SUCCESSFULLY")
    print(f"Results saved to: {results_dir}")
    print(f"Figures generated in: {figures_dir}")
    print(f"Energy Efficiency Gain: {efficiency['energy_efficiency_gain']:.2f}x")
    print(f"SNN Spike Sparsity:     {snn_ops['spike_sparsity_percent']:.2f}%")
    print("=" * 75)


if __name__ == "__main__":
    run_full_suite()
