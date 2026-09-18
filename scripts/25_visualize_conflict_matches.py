from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.home() / "roundabout_ssm"

DATA = ROOT / "data/raw/data"
MATCH_FILE = ROOT / "data/processed/sample_conflict_vehicles.csv"
CONFLICT_FILE = ROOT / "configs/location0_conflict_geometry.json"
ENTRY_FILE = ROOT / "configs/location0_geometry.json"

OUT = ROOT / "outputs/conflict_match_qa.png"

HISTORY_FRAMES = 50
FUTURE_FRAMES = 50


matches = pd.read_csv(MATCH_FILE)

with open(CONFLICT_FILE, "r") as f:
    conflict_geom = json.load(f)

with open(ENTRY_FILE, "r") as f:
    entry_geom = json.load(f)


entries = {
    int(e["entryId"]): e
    for e in conflict_geom["entries"]
}

entry_lines = {
    int(e["id"]): e
    for e in entry_geom["entries"]
}


# ------------------------------------------------------------
# Roundabout circle approximation
# ------------------------------------------------------------

points = np.array([
    entries[i]["conflictPoint"]
    for i in range(4)
])

center = points.mean(axis=0)

radius = np.mean(
    np.sqrt(
        np.sum(
            (points - center) ** 2,
            axis=1
        )
    )
)


# ------------------------------------------------------------
# Select deterministic samples
#
# Pick different TTC quantiles for GO and WAIT so we inspect
# both easy and borderline examples.
# ------------------------------------------------------------

matched = matches[
    matches["leadTrackId"] != -1
].copy()


def pick_quantile_samples(df, decision):
    d = df[
        df["decision"] == decision
    ].sort_values(
        "leadConflictTTC"
    ).reset_index(drop=True)

    qs = [0.20, 0.40, 0.60, 0.80]

    rows = []

    for q in qs:
        idx = int(
            round(
                q * (len(d) - 1)
            )
        )
        rows.append(
            d.iloc[idx]
        )

    return rows


samples = (
    pick_quantile_samples(
        matched, 0
    )
    +
    pick_quantile_samples(
        matched, 1
    )
)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2, 4,
    figsize=(20, 10)
)

axes = axes.flatten()


for ax, sample in zip(
    axes,
    samples
):
    rec = f"{int(sample['recordingId']):02d}"

    ego_id = int(
        sample["trackId"]
    )

    lead_id = int(
        sample["leadTrackId"]
    )

    frame = int(
        sample["currentFrame"]
    )

    entry_id = int(
        sample["entryId"]
    )

    tracks = pd.read_csv(
        DATA / f"{rec}_tracks.csv",
        usecols=[
            "trackId",
            "frame",
            "xCenter",
            "yCenter",
            "xVelocity",
            "yVelocity",
        ]
    )

    ego = tracks[
        tracks["trackId"] == ego_id
    ].sort_values("frame")

    lead = tracks[
        tracks["trackId"] == lead_id
    ].sort_values("frame")


    # current rows
    ego_now = ego[
        ego["frame"] == frame
    ]

    lead_now = lead[
        lead["frame"] == frame
    ]

    if (
        len(ego_now) == 0
        or len(lead_now) == 0
    ):
        continue

    ego_now = ego_now.iloc[0]
    lead_now = lead_now.iloc[0]


    # history / future only for visual QA
    ego_hist = ego[
        (ego["frame"] >= frame - HISTORY_FRAMES)
        &
        (ego["frame"] <= frame)
    ]

    ego_future = ego[
        (ego["frame"] > frame)
        &
        (ego["frame"] <= frame + FUTURE_FRAMES)
    ]

    lead_hist = lead[
        (lead["frame"] >= frame - HISTORY_FRAMES)
        &
        (lead["frame"] <= frame)
    ]

    lead_future = lead[
        (lead["frame"] > frame)
        &
        (lead["frame"] <= frame + FUTURE_FRAMES)
    ]


    # roundabout ring
    theta = np.linspace(
        0,
        2 * np.pi,
        300
    )

    ax.plot(
        center[0] + radius * np.cos(theta),
        center[1] + radius * np.sin(theta),
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="ring approximation"
    )


    # entry line
    e = entry_lines[
        entry_id
    ]

    p1 = e["p1_world"]
    p2 = e["p2_world"]

    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        linewidth=3,
        label="entry line"
    )


    # conflict point
    cp = entries[
        entry_id
    ]["conflictPoint"]

    ax.scatter(
        cp[0],
        cp[1],
        marker="x",
        s=130,
        linewidths=3,
        label="conflict point"
    )


    # ego trajectory
    ax.plot(
        ego_hist["xCenter"],
        ego_hist["yCenter"],
        linewidth=3,
        label="Ego history"
    )

    ax.plot(
        ego_future["xCenter"],
        ego_future["yCenter"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Ego future (QA only)"
    )


    # lead trajectory
    ax.plot(
        lead_hist["xCenter"],
        lead_hist["yCenter"],
        linewidth=3,
        label="Lead history"
    )

    ax.plot(
        lead_future["xCenter"],
        lead_future["yCenter"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Lead future (QA only)"
    )


    # current positions
    ax.scatter(
        ego_now["xCenter"],
        ego_now["yCenter"],
        s=90,
        marker="o"
    )

    ax.scatter(
        lead_now["xCenter"],
        lead_now["yCenter"],
        s=90,
        marker="s"
    )


    # velocity arrows
    ax.arrow(
        ego_now["xCenter"],
        ego_now["yCenter"],
        ego_now["xVelocity"] * 0.5,
        ego_now["yVelocity"] * 0.5,
        head_width=0.6,
        length_includes_head=True
    )

    ax.arrow(
        lead_now["xCenter"],
        lead_now["yCenter"],
        lead_now["xVelocity"] * 0.5,
        lead_now["yVelocity"] * 0.5,
        head_width=0.6,
        length_includes_head=True
    )


    decision = (
        "GO"
        if int(sample["decision"]) == 1
        else "WAIT"
    )


    ax.set_title(
        f"{decision} | Rec {rec} | "
        f"Ego {ego_id} / Lead {lead_id}\n"
        f"TTE(label)={sample['timeToEntry']:.1f}s | "
        f"Lead TTC={sample['leadConflictTTC']:.2f}s"
    )

    ax.set_aspect(
        "equal"
    )

    ax.grid(True)

    # zoom around roundabout
    ax.set_xlim(
        center[0] - radius - 15,
        center[0] + radius + 15
    )

    ax.set_ylim(
        center[1] - radius - 15,
        center[1] + radius + 15
    )


# one shared legend
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4
)

plt.tight_layout(
    rect=[0, 0.07, 1, 1]
)

plt.savefig(
    OUT,
    dpi=180,
    bbox_inches="tight"
)

print("Saved:", OUT)

plt.show()
