from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path.home() / "roundabout_ssm"

OUT = (
    ROOT /
    "outputs/lstm_mamba_trajectory_comparison"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


models = [
    "LSTM",
    "Mamba2",
]

ade = [
    0.8273,
    0.7742,
]

fde = [
    2.0360,
    1.8905,
]


# ============================================================
# ADE
# ============================================================

plt.figure(
    figsize=(7, 5)
)

bars = plt.bar(
    models,
    ade
)

plt.ylabel(
    "ADE [m] ↓"
)

plt.title(
    "4-second Future Trajectory ADE"
)

plt.ylim(
    0,
    0.9
)

plt.grid(
    axis="y",
    alpha=0.3
)

for bar, value in zip(
    bars,
    ade
):
    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.012,
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    OUT /
    "01_lstm_mamba_ade.png",
    dpi=220,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FDE
# ============================================================

plt.figure(
    figsize=(7, 5)
)

bars = plt.bar(
    models,
    fde
)

plt.ylabel(
    "FDE [m] ↓"
)

plt.title(
    "4-second Future Trajectory FDE"
)

plt.ylim(
    0,
    2.2
)

plt.grid(
    axis="y",
    alpha=0.3
)

for bar, value in zip(
    bars,
    fde
):
    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.03,
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    OUT /
    "02_lstm_mamba_fde.png",
    dpi=220,
    bbox_inches="tight"
)

plt.close()


ade_reduction = (
    (ade[0] - ade[1])
    / ade[0]
    * 100
)

fde_reduction = (
    (fde[0] - fde[1])
    / fde[0]
    * 100
)


print("=" * 90)
print("LSTM vs MAMBA2 TRAJECTORY COMPARISON")
print("=" * 90)

print(
    f"LSTM   ADE : {ade[0]:.4f} m"
)

print(
    f"Mamba2 ADE : {ade[1]:.4f} m"
)

print(
    f"ADE reduction : "
    f"{ade_reduction:.2f}%"
)

print()

print(
    f"LSTM   FDE : {fde[0]:.4f} m"
)

print(
    f"Mamba2 FDE : {fde[1]:.4f} m"
)

print(
    f"FDE reduction : "
    f"{fde_reduction:.2f}%"
)

print()

print(
    "Saved:",
    OUT
)

print("=" * 90)
