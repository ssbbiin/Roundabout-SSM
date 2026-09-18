from pathlib import Path
import sys
import time
import csv

import torch

ROOT = Path.home() / "roundabout_ssm"
sys.path.insert(0, str(ROOT))

from models.lstm_baseline import LSTMBaseline
from models.mamba_baseline import MambaBaseline


DEVICE = "cuda"
WARMUP = 50
REPEATS = 200

BATCH_SIZES = [1, 16, 64, 128]


def load_models():

    lstm = LSTMBaseline(
        feature_dim=9,
        step_embed_dim=32,
        hidden_dim=64,
        lstm_layers=2,
        dropout=0.1,
    ).to(DEVICE)

    mamba = MambaBaseline(
        feature_dim=9,
        d_model=64,
        d_state=64,
        num_layers=2,
        dropout=0.1,
    ).to(DEVICE)

    lstm_ckpt = torch.load(
        ROOT / "outputs/lstm_baseline/best.pt",
        map_location=DEVICE,
        weights_only=False
    )

    mamba_ckpt = torch.load(
        ROOT / "outputs/mamba_baseline/best.pt",
        map_location=DEVICE,
        weights_only=False
    )

    lstm.load_state_dict(
        lstm_ckpt["model"]
    )

    mamba.load_state_dict(
        mamba_ckpt["model"]
    )

    lstm.eval()
    mamba.eval()

    return lstm, mamba


@torch.inference_mode()
def benchmark(model, batch_size):

    agents = torch.randn(
        batch_size,
        9,
        50,
        9,
        device=DEVICE
    )

    mask = torch.ones(
        batch_size,
        9,
        device=DEVICE
    )

    entry = torch.randn(
        batch_size,
        2,
        2,
        device=DEVICE
    )

    # warm-up
    for _ in range(WARMUP):
        model(
            agents,
            mask,
            entry
        )

    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    for _ in range(REPEATS):
        model(
            agents,
            mask,
            entry
        )

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    total_samples = (
        batch_size * REPEATS
    )

    latency_batch_ms = (
        elapsed / REPEATS * 1000
    )

    latency_sample_ms = (
        elapsed / total_samples * 1000
    )

    throughput = (
        total_samples / elapsed
    )

    peak_vram_mb = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    return {
        "batch_size":
            batch_size,

        "batch_latency_ms":
            latency_batch_ms,

        "sample_latency_ms":
            latency_sample_ms,

        "throughput":
            throughput,

        "peak_vram_mb":
            peak_vram_mb
    }


lstm, mamba = load_models()

models = {
    "LSTM": lstm,
    "Mamba2": mamba,
}

print("=" * 100)
print("MODEL INFERENCE BENCHMARK")
print("=" * 100)

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print()

rows = []

for name, model in models.items():

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"{name} params: "
        f"{params:,}"
    )

    for bs in BATCH_SIZES:

        result = benchmark(
            model,
            bs
        )

        result["model"] = name
        result["params"] = params

        rows.append(result)

        print(
            f"{name:7s} | "
            f"BS={bs:3d} | "
            f"batch={result['batch_latency_ms']:8.3f} ms | "
            f"sample={result['sample_latency_ms']:8.4f} ms | "
            f"throughput={result['throughput']:9.1f} samples/s | "
            f"VRAM={result['peak_vram_mb']:7.1f} MB"
        )

    print()


out = (
    ROOT /
    "outputs/model_benchmark.csv"
)

with open(
    out,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "params",
            "batch_size",
            "batch_latency_ms",
            "sample_latency_ms",
            "throughput",
            "peak_vram_mb",
        ]
    )

    writer.writeheader()
    writer.writerows(rows)

print("Saved:", out)
print("=" * 100)
