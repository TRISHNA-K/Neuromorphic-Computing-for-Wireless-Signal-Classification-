"""
Evaluation script across SNR sweep (-10 dB to +20 dB) with confusion matrix and classification metrics.
"""

import os
import argparse
from typing import Dict, List, Any
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from src.dataset import WirelessSignalGenerator, WirelessSignalDataset, SUPPORTED_MODULATIONS
from src.models import build_snn_model, build_ann_model
from src.utils import set_seed, save_json, plot_confusion_matrix, get_device


def evaluate_snr_sweep(
    model: torch.nn.Module,
    model_type: str,
    encoder_type: str = "delta",
    snr_levels: List[int] = None,
    samples_per_snr: int = 50,
    sequence_length: int = 128,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Evaluate accuracy across a fine-grained SNR sweep."""
    if snr_levels is None:
        snr_levels = [-10, -6, -2, 2, 6, 10, 14, 18, 20]

    gen = WirelessSignalGenerator(sequence_length=sequence_length, seed=999)
    model.eval()

    acc_per_snr = []
    all_preds = []
    all_targets = []
    all_snrs = []

    for snr in snr_levels:
        X, y, snrs = gen.generate_dataset(
            samples_per_mod_per_snr=samples_per_snr,
            snr_range=(snr, snr, 1),
        )

        dataset = WirelessSignalDataset(
            X=X,
            y=y,
            snrs=snrs,
            encoder_type=encoder_type if model_type == "snn" else None,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

        snr_correct = 0
        snr_total = 0

        with torch.no_grad():
            for inputs, labels, s in loader:
                labels = labels.to(device)

                if model_type == "snn":
                    inputs = inputs.permute(1, 0, 2).to(device)
                    spk_out, _, _ = model(inputs)
                    preds = spk_out.sum(dim=0).argmax(dim=-1)
                else:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    preds = outputs.argmax(dim=-1)

                snr_correct += (preds == labels).sum().item()
                snr_total += labels.size(0)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(labels.cpu().numpy().tolist())
                all_snrs.extend([snr] * labels.size(0))

        snr_acc = float(snr_correct / max(1, snr_total))
        acc_per_snr.append(snr_acc)
        print(f"SNR {snr:+3d} dB -> Accuracy: {snr_acc * 100:.2f}%")

    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)
    cm = confusion_matrix(all_targets_arr, all_preds_arr)
    report = classification_report(
        all_targets_arr, all_preds_arr, target_names=SUPPORTED_MODULATIONS, output_dict=True
    )
    overall_acc = float(np.mean(all_preds_arr == all_targets_arr))

    return {
        "snr_levels": snr_levels,
        "accuracies": acc_per_snr,
        "overall_accuracy": overall_acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate wireless model across SNR sweep.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model_type = ckpt["model_type"]
    arch = ckpt["arch"]
    meta = ckpt.get("meta", {})
    num_classes = meta.get("num_classes", 8)
    encoder = ckpt.get("encoder", "delta")

    print(f"[+] Evaluating {model_type.upper()} ({arch}) from {args.checkpoint}")

    if model_type == "snn":
        model = build_snn_model(arch, in_channels=4, num_classes=num_classes).to(device)
    else:
        model = build_ann_model(arch, in_channels=2, num_classes=num_classes).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    results = evaluate_snr_sweep(
        model=model,
        model_type=model_type,
        encoder_type=encoder,
        device=device,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    res_path = os.path.join(args.output_dir, f"{model_type}_{arch}_snr_results.json")
    save_json(results, res_path)
    print(f"[+] Saved evaluation metrics to {res_path}")

    # Plot confusion matrix
    cm_plot_path = os.path.join(args.output_dir, f"figures/confusion_matrix_{model_type}.png")
    plot_confusion_matrix(
        np.array(results["confusion_matrix"]),
        SUPPORTED_MODULATIONS,
        save_path=cm_plot_path,
        title=f"Confusion Matrix - {model_type.upper()} ({arch})",
    )
    print(f"[+] Saved confusion matrix to {cm_plot_path}")


if __name__ == "__main__":
    main()
