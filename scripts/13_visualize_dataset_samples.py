from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.home() / "roundabout_ssm"
sys.path.insert(0, str(ROOT))

from dataset.roundabout_dataset import RoundaboutDatasetCore


ds = RoundaboutDatasetCore(
    split="train",
    history_frames=50,
    max_neighbors=8,
    neighbor_radius=30.0,
)

# 서로 다른 TTE / 상황을 볼 수 있게 여러 위치 선택
indices = [
    0,
    len(ds) // 4,
    len(ds) // 2,
    len(ds) - 1,
]

fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 12)
)

axes = axes.flat

for ax, idx in zip(axes, indices):

    sample = ds[idx]

    agents = sample["agents"]
    mask = sample["agent_mask"]
    entry_line = sample["entry_line"]
    target = sample["target"]
    meta = sample["metadata"]

    # -----------------------------------------
    # Entry line
    # -----------------------------------------
    ax.plot(
        entry_line[:, 0],
        entry_line[:, 1],
        linewidth=4,
        label="Entry line"
    )

    # -----------------------------------------
    # Ego trajectory
    # -----------------------------------------
    ego = agents[0]

    valid = ego[:, 8] > 0

    ax.plot(
        ego[valid, 0],
        ego[valid, 1],
        linewidth=3,
        label="Ego history"
    )

    ax.scatter(
        ego[-1, 0],
        ego[-1, 1],
        s=100,
        marker="o"
    )

    # -----------------------------------------
    # Neighbor trajectories
    # -----------------------------------------
    neighbor_ids = meta["neighbor_ids"]

    for agent_idx in range(1, 9):

        if mask[agent_idx] == 0:
            continue

        agent = agents[agent_idx]

        valid = agent[:, 8] > 0

        if not np.any(valid):
            continue

        if agent_idx - 1 < len(neighbor_ids):
            neighbor_id = neighbor_ids[agent_idx - 1]
        else:
            neighbor_id = "?"

        ax.plot(
            agent[valid, 0],
            agent[valid, 1],
            linewidth=1.5,
            alpha=0.8,
            label=f"N{agent_idx}: {neighbor_id}"
        )

        ax.scatter(
            agent[valid, 0][-1],
            agent[valid, 1][-1],
            s=40
        )

    # Ego current forward direction
    ax.arrow(
        0,
        0,
        3.0,
        0,
        width=0.05,
        head_width=0.5,
        length_includes_head=True
    )

    decision = (
        "GO"
        if int(target["decision"]) == 1
        else "WAIT"
    )

    ax.set_title(
        f"Index {idx} | "
        f"Rec {meta['recording_id']:02d} "
        f"Track {meta['track_id']} "
        f"Entry {meta['entry_id']}\n"
        f"TTE={float(target['time_to_entry']):.1f}s | "
        f"{decision} | "
        f"Neighbors={int(mask.sum()-1)}"
    )

    ax.set_xlabel("Forward [m]")
    ax.set_ylabel("Left [m]")

    ax.axhline(0, linewidth=0.5)
    ax.axvline(0, linewidth=0.5)

    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend(fontsize=7)

plt.tight_layout()

OUT = ROOT / "outputs/dataset_ego_centric_samples.png"

plt.savefig(
    OUT,
    dpi=200,
    bbox_inches="tight"
)

print(f"Saved: {OUT}")

plt.show()
