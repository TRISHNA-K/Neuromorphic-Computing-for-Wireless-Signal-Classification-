"""
Training script for Spiking Neural Networks (SNNs) and Conventional ANNs on wireless signals.
"""

import os
import argparse
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import snntorch.functional as SF

from src.dataset import get_wireless_dataloaders
from src.models import build_snn_model, build_ann_model
from src.utils import set_seed, save_json, get_device


def train_snn_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train SNN for one epoch using accumulated spike cross entropy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    loss_fn = SF.ce_rate_loss()

    for spikes, labels, _ in loader:
        # spikes shape from loader: (batch, seq_len, 4) -> permute to (seq_len, batch, 4)
        spikes = spikes.permute(1, 0, 2).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        spk_out, mem_out, _ = model(spikes)

        # Loss based on output spike count over time
        loss = loss_fn(spk_out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        # Prediction: argmax of total spikes over time
        pred = spk_out.sum(dim=0).argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / max(1, total),
        "accuracy": (correct / max(1, total)) * 100.0,
    }


def eval_snn(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate SNN on validation/test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    loss_fn = SF.ce_rate_loss()

    with torch.no_grad():
        for spikes, labels, _ in loader:
            spikes = spikes.permute(1, 0, 2).to(device)
            labels = labels.to(device)

            spk_out, mem_out, _ = model(spikes)
            loss = loss_fn(spk_out, labels)

            total_loss += loss.item() * labels.size(0)
            pred = spk_out.sum(dim=0).argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return {
        "loss": total_loss / max(1, total),
        "accuracy": (correct / max(1, total)) * 100.0,
    }


def train_ann_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Train conventional ANN for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels, _ in loader:
        inputs = inputs.to(device)  # (batch, 2, seq_len)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        pred = outputs.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / max(1, total),
        "accuracy": (correct / max(1, total)) * 100.0,
    }


def eval_ann(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate conventional ANN on validation/test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            pred = outputs.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return {
        "loss": total_loss / max(1, total),
        "accuracy": (correct / max(1, total)) * 100.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Train SNN or ANN for wireless modulation classification.")
    parser.add_argument("--model_type", type=str, default="snn", choices=["snn", "ann"])
    parser.add_argument("--arch", type=str, default="spiking_cnn")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--samples_per_mod", type=int, default=60)
    parser.add_argument("--encoder", type=str, default="delta")
    parser.add_argument("--save_path", type=str, default="checkpoints/model.pth")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"[+] Training {args.model_type.upper()} ({args.arch}) on device: {device}")

    # Build Dataloaders
    train_loader, val_loader, test_loader, meta = get_wireless_dataloaders(
        samples_per_mod_per_snr=args.samples_per_mod,
        batch_size=args.batch_size,
        encoder_type=args.encoder if args.model_type == "snn" else None,
        seed=args.seed,
    )
    print(f"[+] Dataset generated: {meta['total_samples']} samples across {meta['num_classes']} classes.")

    if args.model_type == "snn":
        model = build_snn_model(
            args.arch,
            in_channels=4,
            num_classes=meta["num_classes"],
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        model = build_ann_model(
            args.arch,
            in_channels=2,
            num_classes=meta["num_classes"],
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        if args.model_type == "snn":
            train_metrics = train_snn_epoch(model, train_loader, optimizer, device)
            val_metrics = eval_snn(model, val_loader, device)
        else:
            train_metrics = train_ann_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = eval_ann(model, val_loader, criterion, device)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": args.model_type,
                    "arch": args.arch,
                    "encoder": args.encoder if args.model_type == "snn" else None,
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                    "meta": meta,
                },
                args.save_path,
            )

    # Final test evaluation
    if os.path.exists(args.save_path):
        ckpt = torch.load(args.save_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    if args.model_type == "snn":
        test_metrics = eval_snn(model, test_loader, device)
    else:
        test_metrics = eval_ann(model, test_loader, criterion, device)

    print(f"\n[+] Best Val Acc: {best_val_acc:.2f}% | Test Acc: {test_metrics['accuracy']:.2f}%")
    history["test_acc"] = test_metrics["accuracy"]
    history["test_loss"] = test_metrics["loss"]

    save_json(history, args.save_path.replace(".pth", "_history.json"))


if __name__ == "__main__":
    main()
