from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path.home() / "roundabout_ssm"
DATA = ROOT / "data/raw/data"
GEOM = ROOT / "configs/location0_geometry.json"
EVENTS = ROOT / "data/processed/location0_entry_crossings.csv"
OUT = ROOT / "outputs/entry_crossing_qa.png"

with open(GEOM, "r") as f:
    geom = json.load(f)

events = pd.read_csv(EVENTS)

# recording 02에서 entry별 3개씩 고정 샘플
rec_events = events[events["recordingId"] == 2].copy()

rng = np.random.default_rng(42)

samples = []

for entry_id in range(4):
    subset = rec_events[rec_events["entryId"] == entry_id]

    if len(subset) >= 3:
        indices = rng.choice(subset.index, size=3, replace=False)
        samples.extend(subset.loc[indices].to_dict("records"))
    else:
        samples.extend(subset.to_dict("records"))

fig, axes = plt.subplots(4, 3, figsize=(15, 18))

for ax in axes.flat:
    ax.set_visible(False)

for idx, event in enumerate(samples):

    ax = axes.flat[idx]
    ax.set_visible(True)

    rec = f"{int(event['recordingId']):02d}"
    track_id = int(event["trackId"])
    entry_id = int(event["entryId"])
    cross_frame = int(event["crossFrame"])

    tracks = pd.read_csv(
        DATA / f"{rec}_tracks.csv",
        usecols=[
            "trackId",
            "frame",
            "xCenter",
            "yCenter"
        ]
    )

    track = tracks[
        tracks["trackId"] == track_id
    ].sort_values("frame")

    entry = geom["entries"][entry_id]

    p1 = entry["p1_world"]
    p2 = entry["p2_world"]

    # 전체 trajectory
    ax.plot(
        track["xCenter"],
        track["yCenter"],
        linewidth=1.5,
        alpha=0.5,
        label="full trajectory"
    )

    # crossing 전후 ±50 frame
    local = track[
        (track["frame"] >= cross_frame - 50) &
        (track["frame"] <= cross_frame + 50)
    ]

    before = local[local["frame"] < cross_frame]
    after = local[local["frame"] >= cross_frame]

    ax.plot(
        before["xCenter"],
        before["yCenter"],
        linewidth=3,
        label="before"
    )

    ax.plot(
        after["xCenter"],
        after["yCenter"],
        linewidth=3,
        label="after"
    )

    # entry line
    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        linewidth=4,
        label=f"ENTRY {entry_id}"
    )

    # crossing point
    ax.scatter(
        [event["x"]],
        [event["y"]],
        s=80,
        marker="x",
        label="crossing"
    )

    ax.set_title(
        f"Entry {entry_id} | Track {track_id}\n"
        f"speed={event['speed']:.2f} m/s, "
        f"heading={event['heading']:.1f}°"
    )

    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend(fontsize=7)

plt.tight_layout()

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=180, bbox_inches="tight")

print(f"Saved: {OUT}")
plt.show()
