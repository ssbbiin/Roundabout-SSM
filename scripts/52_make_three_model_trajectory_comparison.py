from pathlib import Path
import csv
import matplotlib.pyplot as plt

ROOT = Path.home() / "roundabout_ssm"
OUT = ROOT / "outputs/three_model_trajectory_comparison"
OUT.mkdir(parents=True, exist_ok=True)

models = [
    "LSTM + Interaction",
    "Transformer",
    "Mamba2 + Interaction",
]

ade = [0.8273, 0.9114, 0.7742]
fde = [2.0360, 2.1825, 1.8905]

# Save evaluation results
with open(OUT / "trajectory_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "ADE_m", "FDE_m"])
    for name, a, d in zip(models, ade, fde):
        writer.writerow([name, a, d])


def make_plot(values, metric, filename, ymax):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.bar(
        models,
        values,
        width=0.58
    )

    ax.set_title(
        f"4-second Future Trajectory {metric}",
        fontsize=14
    )

    ax.set_ylabel(f"{metric} [m] (Lower is better)")
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ymax * 0.015,
            f"{value:.4f} m",
            ha="center",
            fontsize=11
        )

    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


make_plot(
    ade,
    "ADE",
    "01_three_model_ade.png",
    1.05
)

make_plot(
    fde,
    "FDE",
    "02_three_model_fde.png",
    2.50
)

print("=" * 75)
print("THREE-MODEL TRAJECTORY COMPARISON")
print("=" * 75)

for name, a, d in zip(models, ade, fde):
    print(f"{name:24s} ADE={a:.4f}m | FDE={d:.4f}m")

print()
print("Saved:", OUT)

for p in sorted(OUT.iterdir()):
    print(" ", p.name)

print("=" * 75)
