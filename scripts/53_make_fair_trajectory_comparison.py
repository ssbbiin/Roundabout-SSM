from pathlib import Path
import csv
import matplotlib.pyplot as plt

ROOT = Path.home() / "roundabout_ssm"
OUT = ROOT / "outputs/fair_trajectory_comparison"
OUT.mkdir(parents=True, exist_ok=True)

models = [
    "LSTM",
    "Transformer",
    "Mamba2",
    "Mamba2 + Interaction",
]

ade = [0.8432, 0.9114, 0.7596, 0.7742]
fde = [2.0394, 2.1825, 1.9009, 1.8905]

with open(OUT / "trajectory_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Interaction", "ADE_m", "FDE_m"])

    for i, name in enumerate(models):
        writer.writerow([
            name,
            "Yes" if i == 3 else "No",
            ade[i],
            fde[i],
        ])


def plot(values, metric, filename, ymax):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(models, values, width=0.6)

    ax.set_title(f"4-second Trajectory Prediction: {metric}")
    ax.set_ylabel(f"{metric} [m] (Lower is better)")
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ymax * 0.015,
            f"{value:.4f}",
            ha="center",
            fontsize=11,
        )

    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


plot(ade, "ADE", "01_fair_ade_comparison.png", 1.05)
plot(fde, "FDE", "02_fair_fde_comparison.png", 2.50)

print("FAIR TRAJECTORY COMPARISON")
for name, a, f in zip(models, ade, fde):
    print(f"{name:24s} ADE={a:.4f} m | FDE={f:.4f} m")

print("Saved:", OUT)
