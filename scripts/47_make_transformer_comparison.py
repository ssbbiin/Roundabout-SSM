from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path.home() / "roundabout_ssm"

OUT = (
    ROOT /
    "outputs/transformer_comparison"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# RESULTS
# ============================================================

models = [
    "LSTM",
    "Transformer",
    "Mamba2",
]

f1 = [
    0.9958,
    0.9922,
    0.9968,
]

tte = [
    0.1165,
    0.1311,
    0.1058,
]

speed = [
    0.3773,
    0.4092,
    0.3453,
]

heading = [
    2.747,
    2.798,
    2.411,
]

params = [
    97670,
    109958,
    108434,
]


results = pd.DataFrame({
    "Model": models,
    "Parameters": params,
    "F1": f1,
    "TTE_MAE": tte,
    "Speed_MAE": speed,
    "Heading_MAE": heading,
})

results.to_csv(
    OUT / "sequence_model_results.csv",
    index=False
)


# ============================================================
# 1. PREDICTION PERFORMANCE
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13, 10)
)

axes[0, 0].bar(
    models,
    f1
)

axes[0, 0].set_title(
    "GO / WAIT Prediction F1"
)

axes[0, 0].set_ylabel(
    "F1 Score ↑"
)

axes[0, 0].set_ylim(
    0.988,
    1.000
)


axes[0, 1].bar(
    models,
    tte
)

axes[0, 1].set_title(
    "Time-to-Entry Prediction"
)

axes[0, 1].set_ylabel(
    "MAE [s] ↓"
)


axes[1, 0].bar(
    models,
    speed
)

axes[1, 0].set_title(
    "Entry Speed Prediction"
)

axes[1, 0].set_ylabel(
    "MAE [m/s] ↓"
)


axes[1, 1].bar(
    models,
    heading
)

axes[1, 1].set_title(
    "Entry Heading Prediction"
)

axes[1, 1].set_ylabel(
    "MAE [deg] ↓"
)


for ax in axes.flat:
    ax.grid(
        axis="y",
        alpha=0.3
    )


plt.tight_layout()

plt.savefig(
    OUT /
    "01_lstm_transformer_mamba_performance.png",
    dpi=220,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 2. PARAMETER COMPARISON
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    models,
    np.array(params) / 1000.0
)

plt.ylabel(
    "Parameters [K]"
)

plt.title(
    "Model Parameter Count"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.savefig(
    OUT /
    "02_parameter_comparison.png",
    dpi=220,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 3. EFFICIENCY
# ============================================================

bench = pd.read_csv(
    ROOT /
    "outputs/sequence_model_benchmark.csv"
)

bs128 = (
    bench[
        bench["batch_size"]
        == 128
    ]
    .set_index("model")
    .loc[models]
)


fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)


axes[0].bar(
    models,
    bs128[
        "sample_latency_ms"
    ]
)

axes[0].set_title(
    "Per-sample Latency (BS=128)"
)

axes[0].set_ylabel(
    "Latency [ms] ↓"
)


axes[1].bar(
    models,
    bs128[
        "throughput"
    ]
)

axes[1].set_title(
    "Throughput (BS=128)"
)

axes[1].set_ylabel(
    "Samples/s ↑"
)


axes[2].bar(
    models,
    bs128[
        "peak_vram_mb"
    ]
)

axes[2].set_title(
    "Peak VRAM (BS=128)"
)

axes[2].set_ylabel(
    "VRAM [MB] ↓"
)


for ax in axes:
    ax.grid(
        axis="y",
        alpha=0.3
    )


plt.tight_layout()

plt.savefig(
    OUT /
    "03_lstm_transformer_mamba_efficiency.png",
    dpi=220,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 4. TRAINING CURVES
# ============================================================

lstm = pd.read_csv(
    ROOT /
    "outputs/lstm_baseline/history.csv"
)

transformer = pd.read_csv(
    ROOT /
    "outputs/transformer_baseline/history.csv"
)

mamba = pd.read_csv(
    ROOT /
    "outputs/mamba_baseline/history.csv"
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5)
)


axes[0].plot(
    lstm["epoch"],
    lstm["val_loss"],
    marker="o",
    markersize=3,
    label="LSTM"
)

axes[0].plot(
    transformer["epoch"],
    transformer["val_loss"],
    marker="o",
    markersize=3,
    label="Transformer"
)

axes[0].plot(
    mamba["epoch"],
    mamba["val_loss"],
    marker="o",
    markersize=3,
    label="Mamba2"
)

axes[0].set_xlabel(
    "Epoch"
)

axes[0].set_ylabel(
    "Validation Loss ↓"
)

axes[0].set_title(
    "Validation Loss"
)

axes[0].grid(
    alpha=0.3
)

axes[0].legend()


axes[1].plot(
    lstm["epoch"],
    lstm["val_tte_mae"],
    marker="o",
    markersize=3,
    label="LSTM"
)

axes[1].plot(
    transformer["epoch"],
    transformer["val_tte_mae"],
    marker="o",
    markersize=3,
    label="Transformer"
)

axes[1].plot(
    mamba["epoch"],
    mamba["val_tte_mae"],
    marker="o",
    markersize=3,
    label="Mamba2"
)

axes[1].set_xlabel(
    "Epoch"
)

axes[1].set_ylabel(
    "TTE MAE [s] ↓"
)

axes[1].set_title(
    "Time-to-Entry Validation Error"
)

axes[1].grid(
    alpha=0.3
)

axes[1].legend()


plt.tight_layout()

plt.savefig(
    OUT /
    "04_lstm_transformer_mamba_training.png",
    dpi=220,
    bbox_inches="tight"
)

plt.close()


print("=" * 100)
print("TRANSFORMER COMPARISON FIGURES GENERATED")
print("=" * 100)

for p in sorted(
    OUT.glob("*")
):
    print(p)

print("=" * 100)
