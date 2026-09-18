from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path.home() / "roundabout_ssm"
OUT = ROOT / "outputs/final_results"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# FINAL MODEL RESULTS
# ============================================================

results = pd.DataFrame([
    {
        "Model": "Kinematic",
        "F1": 0.8919,
        "TTE_MAE": 1.6351,
        "Speed_MAE": 2.1481,
        "Heading_MAE": 15.252,
        "ADE": 3.5826,
        "FDE": 9.1725,
    },
    {
        "Model": "Ego+Map LSTM",
        "F1": 0.9950,
        "TTE_MAE": 0.1234,
        "Speed_MAE": 0.3929,
        "Heading_MAE": 2.705,
        "ADE": np.nan,
        "FDE": np.nan,
    },
    {
        "Model": "Full LSTM",
        "F1": 0.9958,
        "TTE_MAE": 0.1165,
        "Speed_MAE": 0.3773,
        "Heading_MAE": 2.747,
        "ADE": np.nan,
        "FDE": np.nan,
    },
    {
        "Model": "Mamba2 v1",
        "F1": 0.9968,
        "TTE_MAE": 0.1058,
        "Speed_MAE": 0.3453,
        "Heading_MAE": 2.411,
        "ADE": np.nan,
        "FDE": np.nan,
    },
    {
        "Model": "Residual LSTM",
        "F1": 0.9958,
        "TTE_MAE": 0.1136,
        "Speed_MAE": 0.3727,
        "Heading_MAE": 2.711,
        "ADE": np.nan,
        "FDE": np.nan,
    },
    {
        "Model": "Final Mamba2",
        "F1": 0.9971,
        "TTE_MAE": 0.1024,
        "Speed_MAE": 0.3456,
        "Heading_MAE": 2.384,
        "ADE": 0.7742,
        "FDE": 1.8905,
    },
])

results.to_csv(
    OUT / "model_results.csv",
    index=False
)

print("=" * 100)
print("FINAL MODEL RESULTS")
print("=" * 100)
print(results.to_string(index=False))
print()


# ============================================================
# FIGURE 1
# Core prediction comparison
# ============================================================

models = [
    "Kinematic",
    "Ego+Map LSTM",
    "Full LSTM",
    "Mamba2 v1",
    "Final Mamba2",
]

d = (
    results[
        results["Model"].isin(models)
    ]
    .set_index("Model")
    .loc[models]
)


fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)


# F1
axes[0, 0].bar(
    d.index,
    d["F1"]
)

axes[0, 0].set_title(
    "GO / WAIT Prediction F1"
)

axes[0, 0].set_ylabel("F1")

# Zoom because learned models are close
axes[0, 0].set_ylim(
    0.88,
    1.005
)


# TTE
axes[0, 1].bar(
    d.index,
    d["TTE_MAE"]
)

axes[0, 1].set_title(
    "Time-to-Entry Prediction"
)

axes[0, 1].set_ylabel(
    "MAE [s] ↓"
)


# Speed
axes[1, 0].bar(
    d.index,
    d["Speed_MAE"]
)

axes[1, 0].set_title(
    "Entry Speed Prediction"
)

axes[1, 0].set_ylabel(
    "MAE [m/s] ↓"
)


# Heading
axes[1, 1].bar(
    d.index,
    d["Heading_MAE"]
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

    ax.tick_params(
        axis="x",
        rotation=20
    )


plt.tight_layout()

fig1 = (
    OUT /
    "01_core_model_comparison.png"
)

plt.savefig(
    fig1,
    dpi=220,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 2
# Trajectory prediction
# ============================================================

traj_names = [
    "Constant Velocity",
    "Final Mamba2",
]

ade = [
    3.5826,
    0.7742,
]

fde = [
    9.1725,
    1.8905,
]


x = np.arange(
    len(traj_names)
)

width = 0.35


plt.figure(
    figsize=(9, 6)
)

plt.bar(
    x - width / 2,
    ade,
    width,
    label="ADE"
)

plt.bar(
    x + width / 2,
    fde,
    width,
    label="FDE"
)

plt.xticks(
    x,
    traj_names
)

plt.ylabel(
    "Displacement Error [m] ↓"
)

plt.title(
    "4-second Future Trajectory Prediction"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.legend()

fig2 = (
    OUT /
    "02_trajectory_comparison.png"
)

plt.savefig(
    fig2,
    dpi=220,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 3
# LSTM vs Mamba inference efficiency
#
# This is v1 backbone comparison under the benchmarked
# non-fused Mamba2 compatibility path.
# ============================================================

efficiency = pd.DataFrame([
    {
        "Model": "LSTM",
        "Parameters": 97670,
        "Latency_ms_BS1": 0.3992,
        "Throughput_BS128": 17281.0,
        "VRAM_MB_BS128": 350.2,
    },
    {
        "Model": "Mamba2",
        "Parameters": 108434,
        "Latency_ms_BS1": 1.5714,
        "Throughput_BS128": 2567.2,
        "VRAM_MB_BS128": 629.0,
    },
])

efficiency.to_csv(
    OUT / "efficiency_results.csv",
    index=False
)


fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)

axes[0].bar(
    efficiency["Model"],
    efficiency["Latency_ms_BS1"]
)

axes[0].set_title(
    "Single-sample Latency"
)

axes[0].set_ylabel(
    "Latency [ms] ↓"
)

axes[1].bar(
    efficiency["Model"],
    efficiency["Throughput_BS128"]
)

axes[1].set_title(
    "Batch Throughput (BS=128)"
)

axes[1].set_ylabel(
    "samples/s ↑"
)

axes[2].bar(
    efficiency["Model"],
    efficiency["VRAM_MB_BS128"]
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

fig3 = (
    OUT /
    "03_efficiency_comparison.png"
)

plt.savefig(
    fig3,
    dpi=220,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 4
# Training histories
# ============================================================

lstm_history = (
    ROOT /
    "outputs/lstm_baseline/history.csv"
)

mamba_history = (
    ROOT /
    "outputs/mamba_baseline/history.csv"
)


if (
    lstm_history.exists()
    and mamba_history.exists()
):

    lstm = pd.read_csv(
        lstm_history
    )

    mamba = pd.read_csv(
        mamba_history
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

    fig4 = (
        OUT /
        "04_training_curves.png"
    )

    plt.savefig(
        fig4,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close()

else:

    fig4 = None


# ============================================================
# IMPROVEMENT SUMMARY
# ============================================================

lstm = results[
    results["Model"]
    == "Full LSTM"
].iloc[0]

final = results[
    results["Model"]
    == "Final Mamba2"
].iloc[0]


tte_improve = (
    (
        lstm["TTE_MAE"]
        - final["TTE_MAE"]
    )
    /
    lstm["TTE_MAE"]
    * 100
)

speed_improve = (
    (
        lstm["Speed_MAE"]
        - final["Speed_MAE"]
    )
    /
    lstm["Speed_MAE"]
    * 100
)

heading_improve = (
    (
        lstm["Heading_MAE"]
        - final["Heading_MAE"]
    )
    /
    lstm["Heading_MAE"]
    * 100
)

ade_improve = (
    (
        3.5826
        - 0.7742
    )
    /
    3.5826
    * 100
)

fde_improve = (
    (
        9.1725
        - 1.8905
    )
    /
    9.1725
    * 100
)


summary = (
    f"""
FINAL RESULT SUMMARY

Full LSTM -> Final Mamba2
----------------------------------------
F1:
  {lstm['F1']:.4f}
  -> {final['F1']:.4f}

TTE MAE:
  {lstm['TTE_MAE']:.4f} s
  -> {final['TTE_MAE']:.4f} s
  ({tte_improve:.2f}% reduction)

Entry Speed MAE:
  {lstm['Speed_MAE']:.4f} m/s
  -> {final['Speed_MAE']:.4f} m/s
  ({speed_improve:.2f}% reduction)

Entry Heading MAE:
  {lstm['Heading_MAE']:.3f} deg
  -> {final['Heading_MAE']:.3f} deg
  ({heading_improve:.2f}% reduction)


Trajectory:
----------------------------------------
Constant Velocity ADE:
  3.5826 m

Final Mamba2 ADE:
  0.7742 m

ADE reduction:
  {ade_improve:.2f}%


Constant Velocity FDE:
  9.1725 m

Final Mamba2 FDE:
  1.8905 m

FDE reduction:
  {fde_improve:.2f}%


NOTE:
Mamba2 latency benchmark used the non-fused
compatibility execution path
(use_mem_eff_path=False).
"""
)


summary_file = (
    OUT /
    "final_summary.txt"
)

summary_file.write_text(
    summary
)

print(summary)


print("=" * 100)
print("FILES GENERATED")
print("=" * 100)

for p in [
    OUT / "model_results.csv",
    OUT / "efficiency_results.csv",
    fig1,
    fig2,
    fig3,
    fig4,
    summary_file,
]:

    if p is not None:
        print(p)

print("=" * 100)
