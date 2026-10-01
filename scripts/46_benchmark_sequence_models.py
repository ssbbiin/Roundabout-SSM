from pathlib import Path
import sys
import time
import csv
import gc

import torch

ROOT = Path.home() / "roundabout_ssm"
sys.path.insert(0, str(ROOT))

from models.lstm_baseline import LSTMBaseline
from models.transformer_baseline import TransformerBaseline
from models.mamba_baseline import MambaBaseline


DEVICE = "cuda"

WARMUP = 50
REPEATS = 200

BATCH_SIZES = [
    1,
    16,
    64,
    128,
]


# ============================================================
# MODEL BUILDERS
# ============================================================

def build_lstm():

    model = LSTMBaseline(
        feature_dim=9,
        step_embed_dim=32,
        hidden_dim=64,
        lstm_layers=2,
        dropout=0.1,
    ).to(DEVICE)

    ckpt = torch.load(
        ROOT / "outputs/lstm_baseline/best.pt",
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        ckpt["model"]
    )

    model.eval()

    return model


def build_transformer():

    model = TransformerBaseline(
        feature_dim=9,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_history_frames=50,
    ).to(DEVICE)

    ckpt = torch.load(
        ROOT / "outputs/transformer_baseline/best.pt",
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        ckpt["model"]
    )

    model.eval()

    return model


def build_mamba():

    model = MambaBaseline(
        feature_dim=9,
        d_model=64,
        d_state=64,
        num_layers=2,
        dropout=0.1,
    ).to(DEVICE)

    ckpt = torch.load(
        ROOT / "outputs/mamba_baseline/best.pt",
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        ckpt["model"]
    )

    model.eval()

    return model


MODEL_BUILDERS = {
    "LSTM": build_lstm,
    "Transformer": build_transformer,
    "Mamba2": build_mamba,
}


# ============================================================
# BENCHMARK
# ============================================================

@torch.inference_mode()
def benchmark(
    model,
    batch_size
):

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


    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    for _ in range(WARMUP):

        model(
            agents,
            mask,
            entry
        )

    torch.cuda.synchronize()


    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    torch.cuda.reset_peak_memory_stats()


    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    start = time.perf_counter()

    for _ in range(REPEATS):

        model(
            agents,
            mask,
            entry
        )

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )


    total_samples = (
        batch_size
        * REPEATS
    )

    batch_latency_ms = (
        elapsed
        / REPEATS
        * 1000
    )

    sample_latency_ms = (
        elapsed
        / total_samples
        * 1000
    )

    throughput = (
        total_samples
        / elapsed
    )

    peak_vram_mb = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )


    del agents
    del mask
    del entry

    return {
        "batch_size":
            batch_size,

        "batch_latency_ms":
            batch_latency_ms,

        "sample_latency_ms":
            sample_latency_ms,

        "throughput":
            throughput,

        "peak_vram_mb":
            peak_vram_mb,
    }


# ============================================================
# RUN
# ============================================================

print("=" * 115)
print("LSTM vs TRANSFORMER vs MAMBA2 INFERENCE BENCHMARK")
print("=" * 115)

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print(
    "Warmup:",
    WARMUP
)

print(
    "Repeats:",
    REPEATS
)

print()


rows = []


for name, builder in MODEL_BUILDERS.items():

    # --------------------------------------------------------
    # Isolate GPU memory per model
    # --------------------------------------------------------

    gc.collect()
    torch.cuda.empty_cache()

    model = builder()

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"{name} params: "
        f"{params:,}"
    )


    for bs in BATCH_SIZES:

        try:

            result = benchmark(
                model,
                bs
            )

        except torch.OutOfMemoryError:

            print(
                f"{name:11s} | "
                f"BS={bs:3d} | "
                f"CUDA OOM"
            )

            torch.cuda.empty_cache()

            continue


        result["model"] = name
        result["params"] = params

        rows.append(
            result
        )


        print(
            f"{name:11s} | "
            f"BS={bs:3d} | "
            f"batch={result['batch_latency_ms']:8.3f} ms | "
            f"sample={result['sample_latency_ms']:8.4f} ms | "
            f"throughput={result['throughput']:9.1f} samples/s | "
            f"VRAM={result['peak_vram_mb']:7.1f} MB"
        )


    print()

    del model

    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# SAVE
# ============================================================

out = (
    ROOT /
    "outputs/sequence_model_benchmark.csv"
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

    writer.writerows(
        rows
    )


print("=" * 115)
print("Saved:", out)
print("=" * 115)
