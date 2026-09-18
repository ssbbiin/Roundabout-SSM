from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.home() / "roundabout_ssm"

DATA = ROOT / "data/raw/data"
EVENTS_FILE = ROOT / "data/processed/location0_entry_crossings.csv"
ENTRY_GEOM_FILE = ROOT / "configs/location0_geometry.json"

OUT_JSON = ROOT / "configs/location0_conflict_geometry.json"
OUT_PNG = ROOT / "outputs/location0_conflict_geometry.png"

LOOKAHEAD_SEC = 0.4

MOTOR_CLASSES = {
    "car",
    "van",
    "truck",
    "bus",
    "trailer"
}


print("=" * 100)
print("BUILD CONFLICT GEOMETRY")
print("=" * 100)
print(f"Post-entry lookahead: {LOOKAHEAD_SEC:.2f} sec")
print()


events = pd.read_csv(EVENTS_FILE)

with open(ENTRY_GEOM_FILE, "r") as f:
    entry_geom = json.load(f)


all_points = []


# ============================================================
# Collect positions shortly AFTER each vehicle crosses
# its entry line.
# ============================================================

for rec_id, rec_events in events.groupby("recordingId"):

    rec_id = int(rec_id)
    rid = f"{rec_id:02d}"

    print(f"Processing recording {rid} ...")

    meta = pd.read_csv(
        DATA / f"{rid}_recordingMeta.csv"
    ).iloc[0]

    frame_rate = float(meta["frameRate"])

    lookahead_frames = int(
        round(
            LOOKAHEAD_SEC
            * frame_rate
        )
    )

    # Only keep the event fields required for this merge.
    # location0_entry_crossings.csv already contains velocity columns,
    # so keeping them would create xVelocity_x / xVelocity_y suffixes.
    rec_events = rec_events[
        [
            "trackId",
            "entryId",
            "crossFrame"
        ]
    ].copy()

    rec_events["targetFrame"] = (
        rec_events["crossFrame"].astype(int)
        + lookahead_frames
    )

    tracks = pd.read_csv(
        DATA / f"{rid}_tracks.csv",
        usecols=[
            "trackId",
            "frame",
            "xCenter",
            "yCenter",
            "xVelocity",
            "yVelocity",
        ]
    )

    merged = rec_events.merge(
        tracks,
        left_on=[
            "trackId",
            "targetFrame"
        ],
        right_on=[
            "trackId",
            "frame"
        ],
        how="inner"
    )

    for _, row in merged.iterrows():

        vx = float(
            row["xVelocity"]
        )

        vy = float(
            row["yVelocity"]
        )

        speed = np.hypot(
            vx,
            vy
        )

        # stationary/invalid points are not useful
        if speed < 0.5:
            continue

        all_points.append({
            "recordingId":
                rec_id,

            "trackId":
                int(row["trackId"]),

            "entryId":
                int(row["entryId"]),

            "x":
                float(row["xCenter"]),

            "y":
                float(row["yCenter"]),

            "vx":
                vx,

            "vy":
                vy,

            "speed":
                speed
        })


points = pd.DataFrame(
    all_points
)

print()
print(
    "Collected post-entry points:",
    len(points)
)


# ============================================================
# Build static geometry for each entry
# ============================================================

conflict_entries = []


for entry_id in range(4):

    d = points[
        points["entryId"] == entry_id
    ].copy()

    if len(d) == 0:
        raise RuntimeError(
            f"No points for entry {entry_id}"
        )

    # Robust spatial center
    cx = float(
        d["x"].median()
    )

    cy = float(
        d["y"].median()
    )

    # Use normalized velocity vectors to estimate
    # the local direction of circulation.
    speed = np.sqrt(
        d["vx"].to_numpy() ** 2
        +
        d["vy"].to_numpy() ** 2
    )

    ux = (
        d["vx"].to_numpy()
        / speed
    )

    uy = (
        d["vy"].to_numpy()
        / speed
    )

    tx = float(
        np.mean(ux)
    )

    ty = float(
        np.mean(uy)
    )

    tnorm = np.hypot(
        tx,
        ty
    )

    tx /= tnorm
    ty /= tnorm

    tangent_heading_deg = float(
        np.degrees(
            np.arctan2(
                ty,
                tx
            )
        )
    )

    radial_error = np.sqrt(
        (
            d["x"].to_numpy()
            - cx
        ) ** 2
        +
        (
            d["y"].to_numpy()
            - cy
        ) ** 2
    )

    median_spread = float(
        np.median(
            radial_error
        )
    )

    p90_spread = float(
        np.percentile(
            radial_error,
            90
        )
    )

    conflict_entries.append({
        "entryId":
            entry_id,

        "conflictPoint": [
            cx,
            cy
        ],

        "circulationTangent": [
            tx,
            ty
        ],

        "circulationHeadingDeg":
            tangent_heading_deg,

        "samples":
            int(len(d)),

        "medianSpreadMeters":
            median_spread,

        "p90SpreadMeters":
            p90_spread,
    })


result = {
    "locationId": 0,
    "method":
        "median vehicle position after entry-line crossing",

    "lookaheadSec":
        LOOKAHEAD_SEC,

    "entries":
        conflict_entries
}


with open(
    OUT_JSON,
    "w"
) as f:

    json.dump(
        result,
        f,
        indent=2
    )


# ============================================================
# Print statistics
# ============================================================

print()
print("=" * 100)
print("CONFLICT GEOMETRY")
print("=" * 100)

for e in conflict_entries:

    cp = e["conflictPoint"]
    tangent = e[
        "circulationTangent"
    ]

    print(
        f"Entry {e['entryId']} | "
        f"N={e['samples']:5d} | "
        f"point=({cp[0]:7.3f}, {cp[1]:7.3f}) | "
        f"tangent=({tangent[0]:+.3f}, {tangent[1]:+.3f}) | "
        f"heading={e['circulationHeadingDeg']:7.2f} deg | "
        f"median spread={e['medianSpreadMeters']:.2f} m | "
        f"p90={e['p90SpreadMeters']:.2f} m"
    )


# ============================================================
# Visualization on world trajectories
# recording 02
# ============================================================

REC = "02"

tracks = pd.read_csv(
    DATA / f"{REC}_tracks.csv",
    usecols=[
        "trackId",
        "xCenter",
        "yCenter"
    ]
)

track_meta = pd.read_csv(
    DATA / f"{REC}_tracksMeta.csv"
)

motor_ids = set(
    track_meta.loc[
        track_meta["class"].isin(
            MOTOR_CLASSES
        ),
        "trackId"
    ].astype(int)
)

tracks = tracks[
    tracks["trackId"].isin(
        motor_ids
    )
]


plt.figure(
    figsize=(12, 9)
)


# background trajectories
for track_id, g in tracks.groupby(
    "trackId"
):

    if int(track_id) % 3 != 0:
        continue

    plt.plot(
        g["xCenter"],
        g["yCenter"],
        linewidth=0.5,
        alpha=0.25
    )


# Entry lines
for e in entry_geom["entries"]:

    p1 = e["p1_world"]
    p2 = e["p2_world"]

    plt.plot(
        [
            p1[0],
            p2[0]
        ],
        [
            p1[1],
            p2[1]
        ],
        linewidth=3
    )


# Conflict points + tangent direction
for e in conflict_entries:

    x, y = e[
        "conflictPoint"
    ]

    tx, ty = e[
        "circulationTangent"
    ]

    plt.scatter(
        x,
        y,
        s=130,
        marker="x"
    )

    plt.arrow(
        x,
        y,
        tx * 5.0,
        ty * 5.0,
        width=0.10,
        head_width=0.8,
        length_includes_head=True
    )

    plt.text(
        x + 1.0,
        y + 1.0,
        f"CONFLICT {e['entryId']}",
        fontsize=10,
        bbox=dict(
            facecolor="white",
            alpha=0.8
        )
    )


plt.xlabel("x [m]")
plt.ylabel("y [m]")

plt.title(
    "Location 0 - Data-derived Conflict Points"
)

plt.axis("equal")
plt.grid(True)

plt.savefig(
    OUT_PNG,
    dpi=200,
    bbox_inches="tight"
)

print()
print("Saved:")
print(OUT_JSON)
print(OUT_PNG)

plt.show()
