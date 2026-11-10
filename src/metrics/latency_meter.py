"""
Inference latency and throughput benchmarking module for neuromorphic SNNs and conventional ANNs.
Includes:
- Warmup and high-precision timing (mean, median, p95, p99, std, throughput).
- Early Decision / Time-To-Classification analysis across temporal simulation steps.
"""

import time
from typing import Dict, Any, Callable, Optional
import torch
import torch.nn as nn
import numpy as np


class LatencyMeter:
    """Benchmark inference latency and throughput with statistically rigorous repeats."""

    @staticmethod
    def measure_latency(
        model: nn.Module,
        sample_input: torch.Tensor,
        num_warmup: int = 20,
        num_runs: int = 100,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, float]:
        """
        Measure inference latency on single sample or batch.
        Args:
            model: PyTorch model.
            sample_input: Input tensor.
            num_warmup: Number of untimed warmup iterations.
            num_runs: Number of timed runs.
            device: Evaluation device.
        Returns:
            Dict containing mean, median, p95, p99, std latency (ms), and throughput (samples/sec).
        """
        model.eval()
        model.to(device)
        sample_input = sample_input.to(device)

        # Infer batch size
        # SNN input may be (time_steps, batch, channels) or (batch, channels, seq_len)
        if sample_input.dim() == 3:
            # If shape is (T, B, C) where T == 128
            if sample_input.shape[0] > sample_input.shape[1] and sample_input.shape[2] <= 8:
                batch_size = sample_input.shape[1]
            else:
                batch_size = sample_input.shape[0]
        else:
            batch_size = sample_input.shape[0]

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(sample_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = model(sample_input)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)  # ms

        latencies = np.array(latencies)
        mean_lat = float(np.mean(latencies))
        median_lat = float(np.median(latencies))
        p95_lat = float(np.percentile(latencies, 95))
        p99_lat = float(np.percentile(latencies, 99))
        std_lat = float(np.std(latencies))

        # Throughput in samples per second
        per_batch_sec = mean_lat / 1000.0
        throughput = float(batch_size / max(1e-8, per_batch_sec))

        return {
            "mean_latency_ms": mean_lat,
            "median_latency_ms": median_lat,
            "p95_latency_ms": p95_lat,
            "p99_latency_ms": p99_lat,
            "std_latency_ms": std_lat,
            "throughput_samples_per_sec": throughput,
            "batch_size": batch_size,
        }

    @staticmethod
    def evaluate_early_decision_snn(
        model: nn.Module,
        spike_inputs: torch.Tensor,
        confidence_threshold: float = 0.85,
    ) -> Dict[str, Any]:
        """
        Evaluate SNN Time-To-Classification (how many time steps are needed before
        the output spike accumulator reaches high confidence).
        spike_inputs: (time_steps, batch, in_channels)
        """
        model.eval()
        with torch.no_grad():
            spk_out, mem_out, _ = model(spike_inputs)

        time_steps, batch_size, num_classes = spk_out.shape
        # Cumulative spikes over time: (time_steps, batch, num_classes)
        cum_spikes = torch.cumsum(spk_out, dim=0)

        early_step_list = []
        for b in range(batch_size):
            decision_step = time_steps  # default if threshold never reached
            for t in range(time_steps):
                spk_t = cum_spikes[t, b]
                total_spks = spk_t.sum().item()
                if total_spks > 0:
                    probs = spk_t / total_spks
                    top_prob = probs.max().item()
                    if top_prob >= confidence_threshold:
                        decision_step = t + 1
                        break
            early_step_list.append(decision_step)

        avg_steps_needed = float(np.mean(early_step_list))
        median_steps = float(np.median(early_step_list))
        fraction_of_sequence = float(avg_steps_needed / time_steps)

        return {
            "total_time_steps": time_steps,
            "avg_steps_to_decision": avg_steps_needed,
            "median_steps_to_decision": median_steps,
            "fraction_of_sequence_used": fraction_of_sequence,
            "temporal_speedup_potential": float(time_steps / max(1.0, avg_steps_needed)),
        }
