"""
Computational Activity and Energy Proxy Meter for SNN vs ANN comparison.
Implements:
- Synaptic Operations (SOPs) counting for SNNs based on event spikes and layer fan-out.
- Multiply-Accumulate (MACs / FLOPs) counting for ANNs.
- Energy consumption estimation based on 45nm CMOS neuromorphic hardware proxies
  (0.9 pJ / Accumulate for SNN vs 4.6 pJ / MAC for ANN).
- Spike Sparsity and Activity Rate metrics.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import numpy as np


# Standard hardware energy constants (45nm CMOS benchmark, Horowitz ISSCC 2014)
ENERGY_MAC_PJ = 4.6   # 4.6 pJ for 32-bit Multiply-and-Accumulate
ENERGY_AC_PJ = 0.9    # 0.9 pJ for 32-bit Sparse Accumulate (SNN Synaptic Operation)


class EnergyMeter:
    """Profiles computational activity and energy proxies for SNNs and ANNs."""

    @staticmethod
    def calculate_ann_macs_and_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """
        Calculate theoretical MACs and FLOPs for conventional ANN models.
        input_shape: (batch_size, channels, seq_len)
        """
        total_macs = 0
        total_params = 0

        # Create dummy tensor
        dummy = torch.zeros(input_shape)

        # Hook to compute operations
        def conv1d_hook(layer: nn.Conv1d, inp, out):
            nonlocal total_macs
            batch, out_channels, out_len = out.shape
            kernel_size = layer.kernel_size[0]
            in_channels = layer.in_channels
            macs = batch * out_len * out_channels * in_channels * kernel_size
            total_macs += macs

        def linear_hook(layer: nn.Linear, inp, out):
            nonlocal total_macs
            batch = inp[0].shape[0]
            macs = batch * layer.in_features * layer.out_features
            total_macs += macs

        def gru_hook(layer: nn.GRU, inp, out):
            nonlocal total_macs
            batch, seq_len, _ = inp[0].shape
            # GRU has 3 gates per direction and layer: 3 * (in * hidden + hidden * hidden)
            macs = 3 * seq_len * (layer.input_size * layer.hidden_size + layer.hidden_size * layer.hidden_size)
            total_macs += batch * macs

        hooks = []
        for module in model.modules():
            if isinstance(module, nn.Conv1d):
                hooks.append(module.register_forward_hook(conv1d_hook))
            elif isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(linear_hook))
            elif isinstance(module, nn.GRU):
                hooks.append(module.register_forward_hook(gru_hook))

        model.eval()
        with torch.no_grad():
            try:
                model(dummy)
            finally:
                for h in hooks:
                    h.remove()

        batch_size = input_shape[0]
        macs_per_sample = total_macs / max(1, batch_size)
        flops_per_sample = macs_per_sample * 2.0  # 1 MAC = 2 FLOPs

        for p in model.parameters():
            total_params += p.numel()

        # Energy estimation in Joules and picoJoules
        energy_pj = macs_per_sample * ENERGY_MAC_PJ
        energy_joules = energy_pj * 1e-12

        return {
            "macs_per_sample": macs_per_sample,
            "flops_per_sample": flops_per_sample,
            "total_parameters": total_params,
            "energy_proxy_pj": energy_pj,
            "energy_proxy_joules": energy_joules,
            "energy_unit": "45nm CMOS (4.6 pJ / MAC)",
        }

    @staticmethod
    def profile_snn_activity(
        model: nn.Module,
        spike_inputs: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, Any]:
        """
        Profile SNN computational activity: Synaptic Operations (SOPs), spike count, and sparsity.
        spike_inputs: (time_steps, batch_size, in_channels)
        """
        model.eval()
        spike_inputs = spike_inputs.to(device)

        with torch.no_grad():
            spk_out, mem_out, layer_spikes = model(spike_inputs, record_all_spikes=True)

        batch_size = spike_inputs.shape[1]
        time_steps = spike_inputs.shape[0]

        total_sops = 0
        total_spikes_fired = 0
        total_neurons = 0
        layer_breakdown = {}

        # Profile Layer by Layer
        if hasattr(model, "conv1") and "input" in layer_spikes:
            # Presynaptic spikes: input
            in_spikes = layer_spikes["input"]  # (T, B, 4)
            spikes_sum = in_spikes.sum().item()
            # Fan out for Conv1D: kernel_size * out_channels
            fan_out = model.conv1.kernel_size[0] * model.conv1.out_channels
            layer_sops = spikes_sum * fan_out
            total_sops += layer_sops
            total_spikes_fired += spikes_sum
            total_neurons += in_spikes.shape[2] * time_steps * batch_size
            layer_breakdown["conv1"] = {
                "spikes": spikes_sum,
                "fan_out": fan_out,
                "sops": layer_sops,
            }

        if hasattr(model, "conv2") and "layer1" in layer_spikes:
            l1_spikes = layer_spikes["layer1"]  # (T, B, C1)
            spikes_sum = l1_spikes.sum().item()
            fan_out = model.conv2.kernel_size[0] * model.conv2.out_channels
            layer_sops = spikes_sum * fan_out
            total_sops += layer_sops
            total_spikes_fired += spikes_sum
            total_neurons += l1_spikes.shape[2] * time_steps * batch_size
            layer_breakdown["conv2"] = {
                "spikes": spikes_sum,
                "fan_out": fan_out,
                "sops": layer_sops,
            }

        if hasattr(model, "fc1") and "layer2" in layer_spikes:
            l2_spikes = layer_spikes["layer2"]
            spikes_sum = l2_spikes.sum().item()
            fan_out = model.fc1.out_features
            layer_sops = spikes_sum * fan_out
            total_sops += layer_sops
            total_spikes_fired += spikes_sum
            total_neurons += l2_spikes.shape[2] * time_steps * batch_size
            layer_breakdown["fc1"] = {
                "spikes": spikes_sum,
                "fan_out": fan_out,
                "sops": layer_sops,
            }

        if hasattr(model, "fc2") and "layer3" in layer_spikes:
            l3_spikes = layer_spikes["layer3"]
            spikes_sum = l3_spikes.sum().item()
            fan_out = model.fc2.out_features
            layer_sops = spikes_sum * fan_out
            total_sops += layer_sops
            total_spikes_fired += spikes_sum
            total_neurons += l3_spikes.shape[2] * time_steps * batch_size
            layer_breakdown["fc2"] = {
                "spikes": spikes_sum,
                "fan_out": fan_out,
                "sops": layer_sops,
            }

        # Check SpikingMLP structure
        if hasattr(model, "fc1") and "input" in layer_spikes and "layer1" in layer_spikes and "conv1" not in layer_breakdown:
            in_spikes = layer_spikes["input"]
            spikes_sum = in_spikes.sum().item()
            fan_out = model.fc1.out_features
            layer_sops = spikes_sum * fan_out
            total_sops += layer_sops
            total_spikes_fired += spikes_sum
            total_neurons += in_spikes.shape[2] * time_steps * batch_size
            layer_breakdown["fc1"] = {"spikes": spikes_sum, "fan_out": fan_out, "sops": layer_sops}

            l1_spikes = layer_spikes["layer1"]
            spikes_sum = l1_spikes.sum().item()
            fan_out = model.fc2.out_features
            layer_sops = spikes_sum * fan_out
            total_sops += layer_sops
            total_spikes_fired += spikes_sum
            total_neurons += l1_spikes.shape[2] * time_steps * batch_size
            layer_breakdown["fc2"] = {"spikes": spikes_sum, "fan_out": fan_out, "sops": layer_sops}

            if hasattr(model, "fc3") and "layer2" in layer_spikes:
                l2_spikes = layer_spikes["layer2"]
                spikes_sum = l2_spikes.sum().item()
                fan_out = model.fc3.out_features
                layer_sops = spikes_sum * fan_out
                total_sops += layer_sops
                total_spikes_fired += spikes_sum
                total_neurons += l2_spikes.shape[2] * time_steps * batch_size
                layer_breakdown["fc3"] = {"spikes": spikes_sum, "fan_out": fan_out, "sops": layer_sops}

        # Per-sample metrics
        sops_per_sample = total_sops / max(1, batch_size)
        spikes_per_sample = total_spikes_fired / max(1, batch_size)

        # Sparsity / activity factor
        avg_activity_factor = (total_spikes_fired / max(1, total_neurons)) if total_neurons > 0 else 0.0
        spike_sparsity = (1.0 - avg_activity_factor) * 100.0

        # Estimated energy consumption proxy (0.9 pJ per synaptic accumulate operation)
        energy_pj = sops_per_sample * ENERGY_AC_PJ
        energy_joules = energy_pj * 1e-12

        total_params = sum(p.numel() for p in model.parameters())

        return {
            "sops_per_sample": sops_per_sample,
            "spikes_per_sample": spikes_per_sample,
            "activity_factor": avg_activity_factor,
            "spike_sparsity_percent": spike_sparsity,
            "total_parameters": total_params,
            "energy_proxy_pj": energy_pj,
            "energy_proxy_joules": energy_joules,
            "energy_unit": "45nm CMOS (0.9 pJ / SOP Accumulate)",
            "layer_breakdown": layer_breakdown,
        }

    @staticmethod
    def compare_efficiency(ann_metrics: Dict[str, Any], snn_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compute relative efficiency metrics between ANN baseline and SNN."""
        ann_energy = ann_metrics["energy_proxy_pj"]
        snn_energy = snn_metrics["energy_proxy_pj"]
        energy_gain = ann_energy / max(1e-8, snn_energy)

        ann_ops = ann_metrics["macs_per_sample"]
        snn_ops = snn_metrics["sops_per_sample"]
        ops_ratio = ann_ops / max(1e-8, snn_ops)

        return {
            "ann_macs_per_sample": ann_ops,
            "snn_sops_per_sample": snn_ops,
            "operations_ratio (ANN_MACs / SNN_SOPs)": ops_ratio,
            "ann_energy_pj": ann_energy,
            "snn_energy_pj": snn_energy,
            "energy_efficiency_gain": energy_gain,
            "energy_saved_percent": max(0.0, (1.0 - snn_energy / max(1e-8, ann_energy)) * 100.0),
        }
