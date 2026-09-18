from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.home() / "roundabout_ssm"
sys.path.insert(0, str(ROOT))

from dataset.trajectory_dataset import (
    RoundaboutTrajectoryDataset
)


ds = RoundaboutTrajectoryDataset(
    split="train",
    future_sec=4.0,
    future_step_sec=0.2,
)

print("=" * 90)
print("FUTURE TRAJECTORY DATASET TEST")
print("=" * 90)

print("Dataset:", len(ds))
print(
    "Future steps:",
    ds.num_future_steps
)


# Pick examples with different TTE values
df = ds.base.interaction_df

targets = [
    0.4,
    1.6,
    2.4,
    4.0,
]

indices = []

for tte in targets:

    match = np.where(
        np.isclose(
            df[
                "timeToEntry"
            ].to_numpy(),
            tte
        )
    )[0]

    indices.append(
        int(match[0])
    )


fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 12)
)

axes = axes.flatten()


for ax, idx in zip(
    axes,
    indices
):

    s = ds[idx]

    ego = (
        s["agents"][0]
        .numpy()
    )

    history_valid = (
        ego[:, 8] > 0
    )

    future = (
        s[
            "future_trajectory"
        ]
        .numpy()
    )

    future_mask = (
        s[
            "future_mask"
        ]
        .numpy()
        > 0
    )

    entry = (
        s[
            "entry_line"
        ]
        .numpy()
    )


    # --------------------------------------------------
    # Ego history
    # --------------------------------------------------

    ax.plot(
        ego[
            history_valid,
            0
        ],
        ego[
            history_valid,
            1
        ],
        linewidth=3,
        label="Past 2s"
    )


    # --------------------------------------------------
    # Future trajectory GT
    # --------------------------------------------------

    ax.plot(
        future[
            future_mask,
            0
        ],
        future[
            future_mask,
            1
        ],
        marker="o",
        markersize=3,
        linewidth=2,
        label="Future GT 4s"
    )


    # --------------------------------------------------
    # Entry line
    # --------------------------------------------------

    ax.plot(
        entry[:, 0],
        entry[:, 1],
        linewidth=4,
        label="Entry line"
    )


    ax.scatter(
        0,
        0,
        s=100,
        marker="x",
        label="Current Ego"
    )


    # Forward arrow
    ax.arrow(
        0,
        0,
        3,
        0,
        width=0.05,
        head_width=0.5,
        length_includes_head=True
    )


    tte = float(
        s[
            "time_to_entry"
        ]
    )

    decision = (
        "GO"
        if int(
            s["decision"]
        ) == 1
        else "WAIT"
    )


    ax.set_title(
        f"TTE={tte:.1f}s | "
        f"{decision} | "
        f"Future valid="
        f"{int(future_mask.sum())}/20"
    )

    ax.set_xlabel(
        "Forward [m]"
    )

    ax.set_ylabel(
        "Left [m]"
    )

    ax.set_aspect(
        "equal"
    )

    ax.grid(True)

    ax.legend()


plt.tight_layout()

OUT = (
    ROOT /
    "outputs/future_trajectory_targets.png"
)

plt.savefig(
    OUT,
    dpi=180,
    bbox_inches="tight"
)

print()
print(
    "Saved:",
    OUT
)

plt.show()
