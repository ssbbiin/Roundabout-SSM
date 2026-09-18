from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.home() / "roundabout_ssm"
DATA = ROOT / "data/raw/data"
GEOM = ROOT / "configs/location0_geometry.json"
OUT = ROOT / "outputs/location0_world_geometry.png"

RECORDING = "02"

with open(GEOM, "r") as f:
    geom = json.load(f)

tracks = pd.read_csv(
    DATA / f"{RECORDING}_tracks.csv",
    usecols=["trackId", "xCenter", "yCenter"]
)

meta = pd.read_csv(DATA / f"{RECORDING}_tracksMeta.csv")

motor_classes = [
    "car",
    "van",
    "truck",
    "bus",
    "trailer"
]

vehicle_ids = set(
    meta.loc[
        meta["class"].isin(motor_classes),
        "trackId"
    ].tolist()
)

tracks = tracks[tracks["trackId"].isin(vehicle_ids)]

plt.figure(figsize=(12, 9))

# 차량 trajectory
for track_id, group in tracks.groupby("trackId"):
    if track_id % 3 != 0:
        continue

    plt.plot(
        group["xCenter"],
        group["yCenter"],
        linewidth=0.5,
        alpha=0.35
    )

# Entry lines
for entry in geom["entries"]:
    p1 = entry["p1_world"]
    p2 = entry["p2_world"]

    plt.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        linewidth=4
    )

    mx = (p1[0] + p2[0]) / 2
    my = (p1[1] + p2[1]) / 2

    plt.text(
        mx,
        my,
        f"ENTRY {entry['id']}",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8)
    )

plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Location 0 - Vehicle Trajectories + Entry Lines")
plt.axis("equal")
plt.grid(True)

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=200, bbox_inches="tight")

print(f"Saved: {OUT}")

plt.show()
